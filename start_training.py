"""
Quick start training script with smaller model for faster training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm

from model import ImageToSceneModel
from dataset import get_data_loaders, save_vocabulary
from train import Criterion, train_epoch, validate
from utils import AverageMeter, save_checkpoint, clip_gradient


def main():
    # Configuration
    config = {
        'csv_file': 'data/stanford Image Paragraph Captioning dataset/stanford_df_rectified.csv',
        'image_dir': 'data/stanford Image Paragraph Captioning dataset/stanford_img/content/stanford_images',
        'batch_size': 16,  # Smaller batch size for faster training
        'epochs': 10,  # Start with fewer epochs
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'embed_dim': 256,  # Smaller dimensions for faster training
        'decoder_dim': 256,
        'attention_dim': 256,
        'dropout': 0.5,
        'min_word_freq': 3,  # Higher frequency to reduce vocabulary size
        'grad_clip': 5.0,
        'num_workers': 2,  # Fewer workers for stability
        'save_dir': 'checkpoints',
        'log_dir': 'logs',
        'device': 'auto'
    }
    
    print("Starting Image-to-Scene Narration Model Training")
    print("=" * 50)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Set device
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(config['log_dir'])
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab = get_data_loaders(
        config['csv_file'], config['image_dir'], 
        config['batch_size'], config['num_workers'], 
        config['min_word_freq']
    )
    
    # Save vocabulary
    vocab_path = os.path.join(config['save_dir'], 'vocab.pkl')
    save_vocabulary(vocab, vocab_path)
    print(f"Vocabulary saved to {vocab_path}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create model
    model = ImageToSceneModel(
        vocab_size=len(vocab),
        embed_dim=config['embed_dim'],
        decoder_dim=config['decoder_dim'],
        attention_dim=config['attention_dim'],
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Loss and optimizer
    criterion = Criterion()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, device, writer, print_freq=50)
        
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
            'args': config
        }
        
        save_checkpoint(checkpoint, is_best, config['save_dir'])
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if is_best:
            print("New best validation loss!")
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config['save_dir']}")
    
    # Test inference on a sample
    print("\nTesting inference on a sample image...")
    model.eval()
    with torch.no_grad():
        for images, captions, caption_lengths in val_loader:
            sample_image = images[0:1].to(device)
            generated_caption, attention_maps = model.generate_caption(sample_image, vocab, max_length=30)
            
            # Convert to words
            caption_words = []
            for idx in generated_caption:
                if idx in vocab.idx2word:
                    word = vocab.idx2word[idx]
                    if word not in ['<start>', '<end>', '<pad>']:
                        caption_words.append(word)
            
            print(f"Generated caption: {' '.join(caption_words)}")
            break


if __name__ == "__main__":
    main()
