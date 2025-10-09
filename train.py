"""
Training script for Image-to-Scene Narration Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from tqdm import tqdm
import argparse
import json

from model import ImageToSceneModel
from dataset import get_data_loaders, save_vocabulary
from utils import AverageMeter, save_checkpoint, load_checkpoint, clip_gradient


class Criterion(nn.Module):
    """Custom criterion for captioning loss"""
    
    def __init__(self):
        super(Criterion, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <pad> token
    
    def forward(self, predictions, targets, decode_lengths):
        """
        Args:
            predictions: (batch_size, max_decode_length, vocab_size)
            targets: (batch_size, max_caption_length)
            decode_lengths: list of decode lengths
        """
        # Create target tensor for loss calculation
        targets_flat = []
        predictions_flat = []
        
        for i, length in enumerate(decode_lengths):
            # Get predictions and targets for this sequence
            pred_seq = predictions[i, :length]  # (length, vocab_size)
            target_seq = targets[i, 1:length+1]  # Skip <start> token, (length,)
            
            predictions_flat.append(pred_seq)
            targets_flat.append(target_seq)
        
        # Concatenate all sequences
        predictions_flat = torch.cat(predictions_flat, dim=0)  # (total_words, vocab_size)
        targets_flat = torch.cat(targets_flat, dim=0)  # (total_words,)
        
        loss = self.loss_fn(predictions_flat, targets_flat)
        return loss


def train_epoch(train_loader, model, criterion, optimizer, epoch, device, writer, print_freq=100, grad_clip=None):
    """Train for one epoch"""
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    start_time = time.time()
    
    for i, (images, captions, caption_lengths) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data_time.update(time.time() - start_time)
        
        # Move to device
        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)
        
        # Forward pass
        predictions, encoded_captions, decode_lengths, alphas, sort_ind = model(images, captions, caption_lengths)
        
        # Calculate loss
        loss = criterion(predictions, encoded_captions, decode_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        
        optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start_time)
        
        # Log to tensorboard
        if i % print_freq == 0:
            writer.add_scalar('Train/Loss', losses.val, epoch * len(train_loader) + i)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + i)
        
        start_time = time.time()
    
    return losses.avg


def validate(val_loader, model, criterion, device, epoch, writer):
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    
    with torch.no_grad():
        for i, (images, captions, caption_lengths) in enumerate(tqdm(val_loader, desc="Validation")):
            # Move to device
            images = images.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)
            
            # Forward pass
            predictions, encoded_captions, decode_lengths, alphas, sort_ind = model(images, captions, caption_lengths)
            
            # Calculate loss
            loss = criterion(predictions, encoded_captions, decode_lengths)
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', losses.avg, epoch)
    
    return losses.avg


def main():
    parser = argparse.ArgumentParser(description='Train Image-to-Scene Narration Model')
    parser.add_argument('--csv_file', type=str, 
                       default='data/stanford Image Paragraph Captioning dataset/stanford_df_rectified.csv',
                       help='Path to CSV file')
    parser.add_argument('--image_dir', type=str,
                       default='data/stanford Image Paragraph Captioning dataset/stanford_img/content/stanford_images',
                       help='Path to image directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--decoder_dim', type=int, default=512, help='Decoder dimension')
    parser.add_argument('--attention_dim', type=int, default=512, help='Attention dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--min_word_freq', type=int, default=2, help='Minimum word frequency')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab = get_data_loaders(
        args.csv_file, args.image_dir, args.batch_size, args.num_workers, args.min_word_freq
    )
    
    # Save vocabulary
    vocab_path = os.path.join(args.save_dir, 'vocab.pkl')
    save_vocabulary(vocab, vocab_path)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Create model
    model = ImageToSceneModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = Criterion()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, device, writer, grad_clip=args.grad_clip)
        
        # Validate
        val_loss = validate(val_loader, model, criterion, device, epoch, writer)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'vocab_size': len(vocab),
            'args': vars(args)
        }
        
        save_checkpoint(checkpoint, is_best, args.save_dir)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if is_best:
            print("New best validation loss!")
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
