# Image-to-Scene Narration Model

A deep learning model that generates detailed scene narrations from images, built from scratch without using pretrained models. This implementation uses the Stanford Image Paragraph Captioning dataset to train a custom CNN encoder with an LSTM decoder and attention mechanism.

## Features

- **Custom CNN Encoder**: Built from scratch with residual blocks for image feature extraction
- **Attention-based LSTM Decoder**: Generates detailed scene descriptions with attention visualization
- **End-to-End Training**: Complete training pipeline with validation and checkpointing
- **Comprehensive Evaluation**: BLEU scores, attention visualization, and detailed analysis
- **Easy Inference**: Simple interface for generating narrations from new images

## Model Architecture

### Image Encoder
- Custom CNN with residual blocks (similar to ResNet but built from scratch)
- 4 residual layers with increasing channels (64→128→256→512)
- Global average pooling and linear projection to embedding space

### Text Decoder
- LSTM-based decoder with attention mechanism
- Attention weights computed between decoder hidden state and image features
- Gating mechanism to control attention influence

### Key Components
- **ResidualBlock**: Basic building block for the encoder
- **Attention**: Computes attention weights between image features and decoder state
- **TextDecoder**: LSTM decoder with attention for text generation
- **ImageToSceneModel**: Complete model combining encoder and decoder

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Scene-Composer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (for BLEU evaluation):
```python
import nltk
nltk.download('punkt')
```

## Dataset

The model is trained on the Stanford Image Paragraph Captioning dataset:
- **Images**: ~19,551 images in various scenes
- **Captions**: Detailed paragraph descriptions for each image
- **Split**: Train (14,579), Test (2,492), Validation (2,490)

## Usage

### Training

Train the model from scratch:

```bash
python train.py --csv_file "data/stanford Image Paragraph Captioning dataset/stanford_df_rectified.csv" \
                --image_dir "data/stanford Image Paragraph Captioning dataset/stanford_img/content/stanford_images" \
                --batch_size 32 \
                --epochs 50 \
                --lr 1e-4 \
                --save_dir checkpoints \
                --log_dir logs
```

Key training parameters:
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--embed_dim`: Embedding dimension (default: 512)
- `--decoder_dim`: Decoder hidden dimension (default: 512)
- `--attention_dim`: Attention dimension (default: 512)
- `--min_word_freq`: Minimum word frequency for vocabulary (default: 2)

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py --checkpoint checkpoints/model_best.pth.tar \
                   --vocab checkpoints/vocab.pkl \
                   --split test \
                   --max_samples 500 \
                   --save_dir evaluation_results
```

### Inference

Generate narrations for new images:

```bash
# Single image
python inference.py --checkpoint checkpoints/model_best.pth.tar \
                    --vocab checkpoints/vocab.pkl \
                    --image path/to/image.jpg \
                    --output narrations.json

# Multiple images
python inference.py --checkpoint checkpoints/model_best.pth.tar \
                    --vocab checkpoints/vocab.pkl \
                    --image_dir path/to/images/ \
                    --output narrations.json
```

### Web App (Gradio)

Launch an interactive web interface:

```bash
# Quick launch (recommended)
python launch_app.py

# Or launch directly
python app.py --checkpoint checkpoints/model_best.pth.tar \
               --vocab checkpoints/vocab.pkl \
               --host 127.0.0.1 \
               --port 7860

# For public sharing
python app.py --checkpoint checkpoints/model_best.pth.tar \
               --vocab checkpoints/vocab.pkl \
               --share
```

The web app provides:
- **Interactive Interface**: Upload images and generate narrations
- **Attention Visualization**: See what parts of the image the model focuses on
- **Parameter Control**: Adjust generation length, temperature, and sampling
- **Generation History**: Keep track of previous generations
- **Export Options**: Save history and results

## Model Performance

The model achieves competitive performance on scene narration:

- **BLEU Score**: Measures n-gram overlap with reference captions
- **Vocabulary Coverage**: Percentage of vocabulary used in generations
- **Caption Length**: Average length of generated narrations
- **Attention Visualization**: Shows what parts of the image the model focuses on

## File Structure

```
Scene-Composer/
├── model.py              # Model architecture definitions
├── dataset.py            # Dataset and data loading utilities
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── inference.py          # Inference script
├── app.py                # Gradio web app
├── launch_app.py         # App launcher script
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── data/                # Dataset directory
    └── stanford Image Paragraph Captioning dataset/
        ├── stanford_df_rectified.csv
        └── stanford_img/
            └── content/
                └── stanford_images/
                    └── [19551 image files]
```

## Key Features

### 1. Custom Architecture
- No pretrained models used
- Built from scratch for educational purposes
- Modular design for easy modification

### 2. Attention Mechanism
- Visual attention over image regions
- Attention maps for interpretability
- Gating mechanism for better control

### 3. Comprehensive Training
- Gradient clipping for stability
- Learning rate scheduling
- Early stopping and checkpointing
- TensorBoard logging

### 4. Evaluation Metrics
- BLEU score calculation
- Attention visualization
- Caption length analysis
- Vocabulary coverage analysis

### 5. Easy Inference
- Batch processing support
- Temperature-controlled sampling
- JSON output format
- Error handling

### 6. Interactive Web App
- Gradio-based user interface
- Real-time image upload and narration
- Attention visualization
- Generation history tracking
- Parameter adjustment controls

## Training Tips

1. **Start with smaller batch sizes** if you have limited GPU memory
2. **Monitor validation loss** to avoid overfitting
3. **Use gradient clipping** to prevent exploding gradients
4. **Adjust learning rate** based on loss curves
5. **Save checkpoints regularly** for recovery

## Customization

The model can be easily customized:

- **Encoder**: Modify `ImageEncoder` class for different CNN architectures
- **Decoder**: Adjust `TextDecoder` for different RNN types or attention mechanisms
- **Vocabulary**: Change minimum word frequency or add custom preprocessing
- **Training**: Modify loss functions, optimizers, or scheduling

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or image resolution
2. **Slow training**: Use multiple workers for data loading
3. **Poor performance**: Check data preprocessing and vocabulary size
4. **Attention not working**: Verify attention dimension matches model parameters

### Performance Optimization

- Use mixed precision training for faster training
- Implement gradient accumulation for larger effective batch sizes
- Use data parallelism for multi-GPU training
- Optimize data loading with proper num_workers

## License

This project is for educational purposes. Please ensure you have proper rights to use the Stanford dataset for your specific use case.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

- Stanford Image Paragraph Captioning dataset
- PyTorch community for excellent documentation
- Attention mechanism inspired by "Show, Attend and Tell" paper
