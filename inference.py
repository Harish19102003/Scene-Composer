"""
Inference script for generating scene narrations from images
"""

import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import os
import json
from tqdm import tqdm

from model import ImageToSceneModel
from dataset import load_vocabulary
from utils import count_parameters
import torchvision.transforms as transforms


class SceneNarrator:
    """Scene narration inference class"""
    
    def __init__(self, checkpoint_path, vocab_path, device='auto'):
        """
        Initialize the scene narrator
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary file
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load vocabulary
        self.vocab = load_vocabulary(vocab_path)
        print(f"Loaded vocabulary with {len(self.vocab)} words")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        print(f"Loaded model with {count_parameters(self.model):,} parameters")
        
        # Setup image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        model = ImageToSceneModel(
            vocab_size=len(self.vocab),
            embed_dim=checkpoint['args']['embed_dim'],
            decoder_dim=checkpoint['args']['decoder_dim'],
            attention_dim=checkpoint['args']['attention_dim'],
            dropout=checkpoint['args']['dropout']
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            return image_tensor, image
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
    
    def generate_narration(self, image_path, max_length=50, temperature=1.0):
        """
        Generate scene narration for an image
        
        Args:
            image_path: Path to input image
            max_length: Maximum length of generated narration
            temperature: Temperature for sampling (1.0 = deterministic)
        
        Returns:
            narration: Generated text narration
            attention_maps: Attention maps for visualization
        """
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # Generate caption
        with torch.no_grad():
            if temperature == 1.0:
                # Deterministic generation (greedy)
                caption_indices, attention_maps = self.model.generate_caption(
                    image_tensor, self.vocab, max_length
                )
            else:
                # Sampling with temperature
                caption_indices, attention_maps = self._sample_caption(
                    image_tensor, max_length, temperature
                )
        
        # Convert indices to words
        narration_words = []
        for idx in caption_indices:
            if idx in self.vocab.idx2word:
                word = self.vocab.idx2word[idx]
                if word not in ['<start>', '<end>', '<pad>']:
                    narration_words.append(word)
        
        narration = ' '.join(narration_words)
        
        return narration, attention_maps, original_image
    
    def _sample_caption(self, image_tensor, max_length, temperature):
        """Generate caption using sampling with temperature"""
        self.model.eval()
        
        # Encode image
        encoder_out = self.model.encoder(image_tensor)
        encoder_out = encoder_out.unsqueeze(1)
        
        # Initialize decoder
        batch_size = encoder_out.size(0)
        h, c = self.model.decoder.init_hidden_state(encoder_out)
        
        # Start with <start> token
        start_token = self.vocab['<start>']
        word = torch.tensor([[start_token]]).to(self.device)
        
        generated_caption = []
        alphas = []
        
        for _ in range(max_length):
            # Embed word
            embeddings = self.model.decoder.embedding(word)
            
            # Attention
            attention_weighted_encoding, alpha = self.model.decoder.attention(encoder_out, h)
            gate = self.model.decoder.sigmoid(self.model.decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # LSTM step
            h, c = self.model.decoder.decode_step(
                torch.cat([embeddings.squeeze(1), attention_weighted_encoding], dim=1),
                (h, c))
            
            # Predict next word with temperature
            logits = self.model.decoder.fc(h) / temperature
            probs = F.softmax(logits, dim=1)
            predicted_id = torch.multinomial(probs, 1)
            
            # Convert to word
            word = predicted_id
            generated_caption.append(predicted_id.item())
            alphas.append(alpha.cpu().numpy())
            
            # Stop if <end> token
            if predicted_id.item() == self.vocab['<end>']:
                break
        
        return generated_caption, alphas
    
    def batch_narrate(self, image_paths, max_length=50, temperature=1.0):
        """
        Generate narrations for multiple images
        
        Args:
            image_paths: List of image paths
            max_length: Maximum length of generated narration
            temperature: Temperature for sampling
        
        Returns:
            results: List of (image_path, narration, attention_maps) tuples
        """
        results = []
        
        for image_path in tqdm(image_paths, desc="Generating narrations"):
            try:
                narration, attention_maps, original_image = self.generate_narration(
                    image_path, max_length, temperature
                )
                results.append((image_path, narration, attention_maps, original_image))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append((image_path, None, None, None))
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Generate scene narrations from images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--output', type=str, default='narrations.json', help='Output file for results')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum narration length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp'], 
                       help='Image file extensions to process')
    
    args = parser.parse_args()
    
    # Initialize narrator
    narrator = SceneNarrator(args.checkpoint, args.vocab, args.device)
    
    # Collect image paths
    image_paths = []
    
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        for ext in args.extensions:
            image_paths.extend([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) 
                              if f.lower().endswith(ext)])
    else:
        print("Please provide either --image or --image_dir")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Generate narrations
    results = narrator.batch_narrate(image_paths, args.max_length, args.temperature)
    
    # Save results
    output_data = []
    for image_path, narration, attention_maps, original_image in results:
        output_data.append({
            'image_path': image_path,
            'narration': narration,
            'max_length': args.max_length,
            'temperature': args.temperature
        })
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Print sample results
    print("\nSample Results:")
    print("=" * 50)
    for i, (image_path, narration, _, _) in enumerate(results[:5]):
        if narration:
            print(f"\nImage: {os.path.basename(image_path)}")
            print(f"Narration: {narration}")


if __name__ == "__main__":
    main()
