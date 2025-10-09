"""
Evaluation script for Image-to-Scene Narration Model
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import argparse
import os
from tqdm import tqdm
import json

from model import ImageToSceneModel
from dataset import StanfordImageCaptionDataset, collate_fn, load_vocabulary
from utils import calculate_bleu_score, generate_caption_batch, count_parameters, get_model_size
import torchvision.transforms as transforms


def load_model(checkpoint_path, vocab, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = ImageToSceneModel(
        vocab_size=len(vocab),
        embed_dim=checkpoint['args']['embed_dim'],
        decoder_dim=checkpoint['args']['decoder_dim'],
        attention_dim=checkpoint['args']['attention_dim'],
        dropout=checkpoint['args']['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def evaluate_metrics(model, test_loader, vocab, device, max_samples=500):
    """Evaluate model with various metrics"""
    print("Evaluating model...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_attention_maps = []
    
    with torch.no_grad():
        for i, (images, captions, caption_lengths) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if i * images.size(0) >= max_samples:
                break
                
            images = images.to(device)
            
            # Generate captions with attention
            batch_captions = []
            batch_attention = []
            
            for j in range(images.size(0)):
                if i * images.size(0) + j >= max_samples:
                    break
                    
                image = images[j:j+1]
                caption, attention_maps = model.generate_caption(image, vocab, max_length=50)
                batch_captions.append(caption)
                batch_attention.append(attention_maps)
            
            # Store results
            for j in range(len(batch_captions)):
                all_predictions.append(batch_captions[j])
                all_targets.append(captions[j].cpu().numpy())
                all_attention_maps.append(batch_attention[j])
    
    # Calculate BLEU score
    bleu_score = calculate_bleu_score(all_predictions, all_targets, vocab)
    
    # Calculate other metrics
    avg_caption_length = np.mean([len(pred) for pred in all_predictions])
    
    # Calculate vocabulary coverage
    generated_words = set()
    for pred in all_predictions:
        for word_idx in pred:
            if word_idx in vocab.idx2word:
                generated_words.add(vocab.idx2word[word_idx])
    
    vocab_coverage = len(generated_words) / len(vocab)
    
    metrics = {
        'bleu_score': bleu_score,
        'avg_caption_length': avg_caption_length,
        'vocab_coverage': vocab_coverage,
        'num_samples': len(all_predictions)
    }
    
    return metrics, all_predictions, all_targets, all_attention_maps


def visualize_attention(model, image_path, vocab, device, save_path=None):
    """Visualize attention maps for a single image"""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate caption with attention
    model.eval()
    with torch.no_grad():
        caption, attention_maps = model.generate_caption(image_tensor, vocab, max_length=30)
    
    # Convert caption to words
    caption_words = []
    for word_idx in caption:
        if word_idx in vocab.idx2word:
            word = vocab.idx2word[word_idx]
            if word not in ['<start>', '<end>', '<pad>']:
                caption_words.append(word)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Attention Visualization', fontsize=16)
    
    # Show original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Show generated caption
    axes[0, 1].text(0.1, 0.5, ' '.join(caption_words), fontsize=12, wrap=True)
    axes[0, 1].set_title('Generated Caption')
    axes[0, 1].axis('off')
    
    # Show attention maps for first few words
    for i, word in enumerate(caption_words[:4]):
        if i < 4:
            row, col = (i + 2) // 3, (i + 2) % 3
            if i < len(attention_maps):
                attention_map = attention_maps[i].reshape(14, 14)  # Assuming 14x14 feature map
                im = axes[row, col].imshow(attention_map, cmap='hot', interpolation='nearest')
                axes[row, col].set_title(f'Attention: {word}')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return ' '.join(caption_words)


def analyze_predictions(predictions, targets, vocab, save_dir):
    """Analyze prediction patterns and save results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert predictions and targets to text
    pred_texts = []
    target_texts = []
    
    for pred, target in zip(predictions, targets):
        pred_words = [vocab.idx2word[idx] for idx in pred if idx in vocab.idx2word and vocab.idx2word[idx] not in ['<start>', '<end>', '<pad>']]
        target_words = [vocab.idx2word[idx] for idx in target if idx in vocab.idx2word and vocab.idx2word[idx] not in ['<start>', '<end>', '<pad>']]
        
        pred_texts.append(' '.join(pred_words))
        target_texts.append(' '.join(target_words))
    
    # Save sample predictions
    with open(os.path.join(save_dir, 'sample_predictions.txt'), 'w') as f:
        f.write("Sample Predictions vs Targets:\n")
        f.write("=" * 50 + "\n\n")
        
        for i in range(min(50, len(pred_texts))):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Target: {target_texts[i]}\n")
            f.write(f"Predicted: {pred_texts[i]}\n")
            f.write("-" * 30 + "\n\n")
    
    # Analyze caption lengths
    pred_lengths = [len(text.split()) for text in pred_texts]
    target_lengths = [len(text.split()) for text in target_texts]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(pred_lengths, bins=20, alpha=0.7, label='Predicted', color='blue')
    plt.hist(target_lengths, bins=20, alpha=0.7, label='Target', color='red')
    plt.xlabel('Caption Length (words)')
    plt.ylabel('Frequency')
    plt.title('Caption Length Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(target_lengths, pred_lengths, alpha=0.5)
    plt.plot([min(target_lengths), max(target_lengths)], [min(target_lengths), max(target_lengths)], 'r--')
    plt.xlabel('Target Length')
    plt.ylabel('Predicted Length')
    plt.title('Length Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'caption_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis results saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Image-to-Scene Narration Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--csv_file', type=str, 
                       default='data/stanford Image Paragraph Captioning dataset/stanford_df_rectified.csv',
                       help='Path to CSV file')
    parser.add_argument('--image_dir', type=str,
                       default='data/stanford Image Paragraph Captioning dataset/stanford_img/content/stanford_images',
                       help='Path to image directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=500, help='Maximum samples to evaluate')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'val'], help='Split to evaluate on')
    parser.add_argument('--save_dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--visualize_attention', type=str, default=None, help='Path to image for attention visualization')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load vocabulary
    vocab = load_vocabulary(args.vocab)
    print(f"Loaded vocabulary with {len(vocab)} words")
    
    # Load model
    model = load_model(args.checkpoint, vocab, device)
    print(f"Loaded model with {count_parameters(model):,} parameters")
    print(f"Model size: {get_model_size(model):.2f} MB")
    
    # Create test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = StanfordImageCaptionDataset(
        args.csv_file, args.image_dir, vocab=vocab, transform=transform, split=args.split
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, collate_fn=collate_fn
    )
    
    # Evaluate model
    metrics, predictions, targets, attention_maps = evaluate_metrics(
        model, test_loader, vocab, device, args.max_samples
    )
    
    # Print metrics
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save metrics
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Analyze predictions
    analyze_predictions(predictions, targets, vocab, args.save_dir)
    
    # Visualize attention if image provided
    if args.visualize_attention:
        caption = visualize_attention(
            model, args.visualize_attention, vocab, device,
            os.path.join(args.save_dir, 'attention_visualization.png')
        )
        print(f"Generated caption: {caption}")
    
    print(f"\nEvaluation completed! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
