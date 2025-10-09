"""
Dataset and DataLoader for Stanford Image Paragraph Captioning Dataset
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import re
from collections import Counter
import pickle


class Vocabulary:
    """Vocabulary class for text processing"""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # Add special tokens
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)


def build_vocabulary(paragraphs, min_word_freq=2):
    """Build vocabulary from paragraphs"""
    print("Building vocabulary...")
    
    # Count word frequencies
    word_counts = Counter()
    for paragraph in paragraphs:
        # Clean and tokenize
        words = clean_text(paragraph).split()
        word_counts.update(words)
    
    # Create vocabulary
    vocab = Vocabulary()
    
    # Add words that meet minimum frequency threshold
    for word, count in word_counts.items():
        if count >= min_word_freq:
            vocab.add_word(word)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total unique words: {len(word_counts)}")
    print(f"Words with freq >= {min_word_freq}: {len(vocab) - 4}")  # -4 for special tokens
    
    return vocab


def clean_text(text):
    """Clean and normalize text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


class StanfordImageCaptionDataset(Dataset):
    """Dataset class for Stanford Image Paragraph Captioning"""
    
    def __init__(self, csv_file, image_dir, vocab=None, transform=None, split='train', min_word_freq=2):
        """
        Args:
            csv_file: Path to the CSV file with annotations
            image_dir: Directory with all the images
            vocab: Vocabulary object (if None, will build from data)
            transform: Optional transform to be applied on images
            split: 'train', 'test', or 'val'
            min_word_freq: Minimum word frequency for vocabulary
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.split = split
        
        # Filter by split
        if split == 'train':
            self.df = self.df[self.df['train'] == True]
        elif split == 'test':
            self.df = self.df[self.df['test'] == True]
        elif split == 'val':
            self.df = self.df[self.df['val'] == True]
        
        print(f"Loaded {len(self.df)} samples for {split} split")
        
        # Clean paragraphs
        self.paragraphs = [clean_text(p) for p in self.df['Paragraph'].tolist()]
        
        # Build or use provided vocabulary
        if vocab is None:
            self.vocab = build_vocabulary(self.paragraphs, min_word_freq)
        else:
            self.vocab = vocab
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and load image
        img_name = self.df.iloc[idx]['Image_name']
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get paragraph and encode
        paragraph = self.paragraphs[idx]
        encoded_paragraph = self.encode_paragraph(paragraph)
        
        return image, encoded_paragraph, len(encoded_paragraph)
    
    def encode_paragraph(self, paragraph):
        """Encode paragraph to list of word indices"""
        words = paragraph.split()
        encoded = [self.vocab('<start>')]
        encoded.extend([self.vocab(word) for word in words])
        encoded.append(self.vocab('<end>'))
        return encoded
    
    def decode_paragraph(self, encoded_paragraph):
        """Decode list of word indices to paragraph"""
        words = [self.vocab.idx2word[idx] for idx in encoded_paragraph]
        return ' '.join(words)


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    # Sort batch by caption length (descending)
    batch.sort(key=lambda x: x[2], reverse=True)
    
    images, captions, lengths = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Pad captions
    max_length = max(lengths)
    padded_captions = torch.zeros(len(captions), max_length, dtype=torch.long)
    
    for i, caption in enumerate(captions):
        padded_captions[i, :len(caption)] = torch.tensor(caption)
    
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return images, padded_captions, lengths


def get_data_loaders(csv_file, image_dir, batch_size=32, num_workers=4, min_word_freq=2):
    """Get train, validation, and test data loaders"""
    
    # Image transforms
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = StanfordImageCaptionDataset(
        csv_file, image_dir, vocab=None, transform=transform_train, 
        split='train', min_word_freq=min_word_freq
    )
    
    val_dataset = StanfordImageCaptionDataset(
        csv_file, image_dir, vocab=train_dataset.vocab, transform=transform_val, 
        split='val', min_word_freq=min_word_freq
    )
    
    test_dataset = StanfordImageCaptionDataset(
        csv_file, image_dir, vocab=train_dataset.vocab, transform=transform_val, 
        split='test', min_word_freq=min_word_freq
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.vocab


def save_vocabulary(vocab, filepath):
    """Save vocabulary to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocabulary(filepath):
    """Load vocabulary from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Test the dataset
    csv_file = "data/stanford Image Paragraph Captioning dataset/stanford_df_rectified.csv"
    image_dir = "data/stanford Image Paragraph Captioning dataset/stanford_img/content/stanford_images"
    
    print("Testing dataset...")
    train_loader, val_loader, test_loader, vocab = get_data_loaders(
        csv_file, image_dir, batch_size=4, num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Test a batch
    for batch_idx, (images, captions, lengths) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Captions shape: {captions.shape}")
        print(f"  Lengths: {lengths}")
        # Create a temporary dataset to use the decode_paragraph method
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        temp_dataset = StanfordImageCaptionDataset(
            csv_file, image_dir, vocab=vocab, transform=transform_val, split='train'
        )
        print(f"  Sample caption: {temp_dataset.decode_paragraph(captions[0].tolist())}")
        break
