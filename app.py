"""
Gradio Web App for Image-to-Scene Narration Model
"""

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import json
from datetime import datetime

from model import ImageToSceneModel
from dataset import load_vocabulary
from utils import count_parameters
import torchvision.transforms as transforms


class SceneNarratorApp:
    """Gradio app for scene narration"""
    
    def __init__(self, checkpoint_path, vocab_path, device='auto'):
        """Initialize the app with model and vocabulary"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
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
        
        # Initialize generation history
        self.generation_history = []
    
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
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        if image is None:
            return None
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def generate_narration(self, image, max_length, temperature, use_sampling):
        """Generate scene narration for an image"""
        if image is None:
            return "Please upload an image first.", None
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Generate caption
            with torch.no_grad():
                if use_sampling and temperature > 1.0:
                    caption_indices, attention_maps = self._sample_caption(
                        image_tensor, max_length, temperature
                    )
                else:
                    caption_indices, attention_maps = self.model.generate_caption(
                        image_tensor, self.vocab, max_length
                    )
            
            # Convert indices to words
            narration_words = []
            for idx in caption_indices:
                if idx in self.vocab.idx2word:
                    word = self.vocab.idx2word[idx]
                    if word not in ['<start>', '<end>', '<pad>']:
                        narration_words.append(word)
            
            narration = ' '.join(narration_words)
            
            # Create attention visualization
            attention_vis = self._create_attention_visualization(image, attention_maps, narration_words)
            
            # Save to history
            self.generation_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'narration': narration,
                'max_length': max_length,
                'temperature': temperature,
                'use_sampling': use_sampling
            })
            
            return narration, attention_vis
            
        except Exception as e:
            return f"Error generating narration: {str(e)}", None
    
    def _sample_caption(self, image_tensor, max_length, temperature, repetition_penalty=1.2, no_repeat_ngram_size=3):
        """Generate caption using sampling with temperature"""
        # Encode image
        encoder_out = self.model.encoder(image_tensor)
        encoder_out = encoder_out.unsqueeze(1)
        
        # Initialize decoder
        batch_size = encoder_out.size(0)
        h, c = self.model.decoder.init_hidden_state(encoder_out)
        
        # Start with <start> token
        start_token = self.vocab.word2idx['<start>']
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
            
            # Predict next word with temperature + repetition controls
            logits = self.model.decoder.fc(h) / max(temperature, 1e-5)
            # Immediate repeat block
            if len(generated_caption) > 0:
                last_token = generated_caption[-1]
                logits[0, last_token] = logits[0, last_token] - 1e9
            # Repetition penalty
            if len(generated_caption) > 0:
                with torch.no_grad():
                    unique_tokens = set(generated_caption)
                for t in unique_tokens:
                    logits[0, t] = logits[0, t] / repetition_penalty
            # No-repeat n-gram (simple mask for last n-1 prefix)
            if no_repeat_ngram_size > 1 and len(generated_caption) >= no_repeat_ngram_size - 1:
                prefix = tuple(generated_caption[-(no_repeat_ngram_size - 1):])
                # Basic heuristic: if the predicted token would recreate the recent n-gram start, downweight
                # (Full tracking of all n-grams omitted for simplicity.)
                # We can lightly penalize common small tokens to diversify
                for t in prefix:
                    logits[0, t] = logits[0, t] - 5.0
            probs = F.softmax(logits, dim=1)
            predicted_id = torch.multinomial(probs, 1)
            
            # Convert to word
            word = predicted_id
            generated_caption.append(predicted_id.item())
            alphas.append(alpha.cpu().numpy())
            
            # Stop if <end> token
            if predicted_id.item() == self.vocab.word2idx['<end>']:
                break
        
        return generated_caption, alphas
    
    def _create_attention_visualization(self, original_image, attention_maps, caption_words):
        """Create attention visualization"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            
            # Create figure
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Attention Visualization', fontsize=16)
            
            # Show original image
            axes[0, 0].imshow(original_image)
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
            
            # Convert to numpy array for Gradio
            fig.canvas.draw()
            vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return vis_array
            
        except Exception as e:
            print(f"Error creating attention visualization: {e}")
            return None
    
    def get_history(self):
        """Get generation history"""
        if not self.generation_history:
            return "No generations yet."
        
        history_text = "Generation History:\n" + "="*50 + "\n\n"
        for i, entry in enumerate(self.generation_history[-10:], 1):  # Show last 10
            history_text += f"{i}. {entry['timestamp']}\n"
            history_text += f"   Narration: {entry['narration']}\n"
            history_text += f"   Settings: Length={entry['max_length']}, Temp={entry['temperature']}, Sampling={entry['use_sampling']}\n"
            history_text += "-" * 30 + "\n\n"
        
        return history_text
    
    def clear_history(self):
        """Clear generation history"""
        self.generation_history = []
        return "History cleared."
    
    def save_history(self):
        """Save generation history to file"""
        if not self.generation_history:
            return "No history to save."
        
        filename = f"generation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.generation_history, f, indent=2)
        
        return f"History saved to {filename}"


def create_gradio_app(checkpoint_path, vocab_path):
    """Create and launch Gradio app"""
    
    # Initialize the narrator
    narrator = SceneNarratorApp(checkpoint_path, vocab_path)
    
    # Create Gradio interface
    with gr.Blocks(title="Image-to-Scene Narration", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🖼️ Image-to-Scene Narration Model
        
        Upload an image and generate a detailed scene narration using our custom-built model!
        The model uses attention mechanisms to focus on different parts of the image while generating descriptions.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input image
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400
                )
                
                # Generation parameters
                with gr.Accordion("Generation Parameters", open=False):
                    max_length = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Maximum Length"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature (for sampling)"
                    )
                    
                    use_sampling = gr.Checkbox(
                        label="Use Sampling (instead of greedy)",
                        value=False
                    )
                
                # Generate button
                generate_btn = gr.Button("Generate Narration", variant="primary", size="lg")
                
                # Clear button
                clear_btn = gr.Button("Clear", variant="secondary")
            
            with gr.Column(scale=1):
                # Output narration
                output_text = gr.Textbox(
                    label="Generated Narration",
                    lines=8,
                    max_lines=15,
                    show_copy_button=True
                )
                
                # Attention visualization
                attention_vis = gr.Image(
                    label="Attention Visualization",
                    height=400
                )
        
        # History section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Generation History")
                history_text = gr.Textbox(
                    label="History",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
                
                with gr.Row():
                    refresh_history_btn = gr.Button("Refresh History")
                    clear_history_btn = gr.Button("Clear History")
                    save_history_btn = gr.Button("Save History")
        
        # Event handlers
        generate_btn.click(
            fn=narrator.generate_narration,
            inputs=[input_image, max_length, temperature, use_sampling],
            outputs=[output_text, attention_vis]
        )
        
        clear_btn.click(
            fn=lambda: (None, "", None),
            outputs=[input_image, output_text, attention_vis]
        )
        
        refresh_history_btn.click(
            fn=narrator.get_history,
            outputs=[history_text]
        )
        
        clear_history_btn.click(
            fn=narrator.clear_history,
            outputs=[history_text]
        )
        
        save_history_btn.click(
            fn=narrator.save_history,
            outputs=[history_text]
        )
        
        # Load initial history
        app.load(
            fn=narrator.get_history,
            outputs=[history_text]
        )
        
        # Examples
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                ["examples/example1.jpg", 50, 1.0, False],
                ["examples/example2.jpg", 30, 1.2, True],
                ["examples/example3.jpg", 40, 0.8, False],
            ],
            inputs=[input_image, max_length, temperature, use_sampling],
            label="Try these examples (if available)"
        )
    
    return app


def main():
    """Main function to run the app"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Gradio app for Image-to-Scene Narration')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pth.tar',
                       help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='checkpoints/vocab.pkl',
                       help='Path to vocabulary file')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to run the app on')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run the app on')
    parser.add_argument('--share', action='store_true',
                       help='Create a public link')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        print("Please train a model first or provide the correct path.")
        return
    
    if not os.path.exists(args.vocab):
        print(f"Error: Vocabulary file not found: {args.vocab}")
        print("Please train a model first or provide the correct path.")
        return
    
    # Create and launch app
    app = create_gradio_app(args.checkpoint, args.vocab)
    
    print(f"Launching Gradio app on http://{args.host}:{args.port}")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
