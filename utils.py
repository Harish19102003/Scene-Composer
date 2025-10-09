"""
Utility functions for training and evaluation
"""

import torch
import torch.nn as nn
import numpy as np
import os
import shutil
from collections import defaultdict


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    """Save checkpoint"""
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_dir, 'model_best.pth.tar'))


def load_checkpoint(filepath):
    """Load checkpoint"""
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        return checkpoint
    else:
        raise FileNotFoundError(f"No checkpoint found at {filepath}")


def clip_gradient(optimizer, grad_clip):
    """Clip gradients computed during backpropagation to avoid explosion of gradients."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def accuracy(scores, targets, k):
    """Compute top-k accuracy"""
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def adjust_learning_rate(optimizer, shrink_factor):
    """Shrink learning rate by a specified factor"""
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def save_attention_maps(images, captions, alphas, vocab, save_dir, epoch, batch_idx):
    """Save attention maps for visualization"""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(4, images.size(0))):  # Save first 4 images
        # Get image and attention weights
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Get caption
        caption = captions[i].cpu().numpy()
        caption_words = [vocab.idx2word[idx] for idx in caption if idx != 0]  # Remove padding
        
        # Get attention weights
        attention_weights = alphas[i].cpu().numpy()  # (seq_len, num_pixels)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Epoch {epoch}, Batch {batch_idx}, Image {i}')
        
        # Show original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Show caption
        axes[0, 1].text(0.1, 0.5, ' '.join(caption_words), fontsize=10, wrap=True)
        axes[0, 1].set_title('Generated Caption')
        axes[0, 1].axis('off')
        
        # Show attention maps for first few words
        for j, word in enumerate(caption_words[:2]):
            if j < 2:
                row, col = 1, j
                if j < attention_weights.shape[0]:
                    attention_map = attention_weights[j].reshape(14, 14)  # Assuming 14x14 feature map
                    im = axes[row, col].imshow(attention_map, cmap='hot', interpolation='nearest')
                    axes[row, col].set_title(f'Attention: {word}')
                    axes[row, col].axis('off')
                    plt.colorbar(im, ax=axes[row, col])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'attention_epoch_{epoch}_batch_{batch_idx}_img_{i}.png'))
        plt.close()


def calculate_bleu_score(predictions, targets, vocab):
    """Calculate BLEU score for generated captions"""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    
    for pred, target in zip(predictions, targets):
        # Convert indices to words
        pred_words = [vocab.idx2word[idx] for idx in pred if idx not in [vocab.word2idx['<start>'], vocab.word2idx['<end>'], vocab.word2idx['<pad>']]]
        target_words = [vocab.idx2word[idx] for idx in target if idx not in [vocab.word2idx['<start>'], vocab.word2idx['<end>'], vocab.word2idx['<pad>']]]
        
        if len(pred_words) > 0 and len(target_words) > 0:
            score = sentence_bleu([target_words], pred_words, smoothing_function=smoothie)
            bleu_scores.append(score)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0


def generate_caption_batch(model, images, vocab, max_length=50, device='cpu'):
    """Generate captions for a batch of images"""
    model.eval()
    captions = []
    
    with torch.no_grad():
        for i in range(images.size(0)):
            image = images[i:i+1]  # Keep batch dimension
            caption, _ = model.generate_caption(image, vocab, max_length)
            captions.append(caption)
    
    return captions


def evaluate_model(model, test_loader, vocab, device, max_samples=100):
    """Evaluate model on test set"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, (images, captions, caption_lengths) in enumerate(test_loader):
            if i * images.size(0) >= max_samples:
                break
                
            images = images.to(device)
            
            # Generate captions
            predictions = generate_caption_batch(model, images, vocab, device=device)
            
            # Store results
            for j in range(images.size(0)):
                if i * images.size(0) + j >= max_samples:
                    break
                    
                all_predictions.append(predictions[j])
                all_targets.append(captions[j].cpu().numpy())
    
    # Calculate BLEU score
    bleu_score = calculate_bleu_score(all_predictions, all_targets, vocab)
    
    return bleu_score, all_predictions, all_targets


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """Get the size of a model in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
