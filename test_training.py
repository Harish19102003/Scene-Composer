"""
Quick test script to verify the model works with the dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ImageToSceneModel
from dataset import get_data_loaders
from utils import AverageMeter
import time

def test_model():
    """Test the model with a small batch"""
    print("Testing model with dataset...")
    
    # Get data loaders
    csv_file = "data/stanford Image Paragraph Captioning dataset/stanford_df_rectified.csv"
    image_dir = "data/stanford Image Paragraph Captioning dataset/stanford_img/content/stanford_images"
    
    train_loader, val_loader, test_loader, vocab = get_data_loaders(
        csv_file, image_dir, batch_size=4, num_workers=0, min_word_freq=2
    )
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ImageToSceneModel(
        vocab_size=len(vocab),
        embed_dim=256,  # Smaller for testing
        decoder_dim=256,
        attention_dim=256,
        dropout=0.5
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for batch_idx, (images, captions, caption_lengths) in enumerate(train_loader):
        print(f"\nTesting batch {batch_idx}")
        print(f"Images shape: {images.shape}")
        print(f"Captions shape: {captions.shape}")
        print(f"Caption lengths: {caption_lengths}")
        
        # Move to device
        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)
        
        # Forward pass
        start_time = time.time()
        predictions, encoded_captions, decode_lengths, alphas, sort_ind = model(images, captions, caption_lengths)
        forward_time = time.time() - start_time
        
        print(f"Forward pass time: {forward_time:.3f}s")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Decode lengths: {decode_lengths}")
        print(f"Alphas shape: {alphas.shape}")
        
        # Calculate loss using the custom criterion from train.py
        from train import Criterion
        criterion_custom = Criterion()
        loss = criterion_custom(predictions, encoded_captions, decode_lengths)
        
        print(f"Loss: {loss.item():.4f}")
        
        # Test backward pass
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        print(f"Backward pass time: {backward_time:.3f}s")
        print("Model test passed!")
        
        break  # Only test one batch
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    
    with torch.no_grad():
        # Get a sample image
        for images, captions, caption_lengths in train_loader:
            sample_image = images[0:1].to(device)
            
            # Generate caption
            start_time = time.time()
            generated_caption, attention_maps = model.generate_caption(sample_image, vocab, max_length=20)
            inference_time = time.time() - start_time
            
            print(f"Inference time: {inference_time:.3f}s")
            print(f"Generated caption indices: {generated_caption}")
            
            # Convert to words
            caption_words = []
            for idx in generated_caption:
                if idx in vocab.idx2word:
                    word = vocab.idx2word[idx]
                    if word not in ['<start>', '<end>', '<pad>']:
                        caption_words.append(word)
            
            print(f"Generated caption: {' '.join(caption_words)}")
            print("Inference test passed!")
            break
    
    print("\nAll tests passed! Model is ready for training.")

if __name__ == "__main__":
    test_model()
