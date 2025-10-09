"""
Image-to-Scene Narration Model
Built from scratch without pretrained models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImageEncoder(nn.Module):
    """
    Custom CNN encoder for image feature extraction
    Built from scratch without pretrained weights
    """
    
    def __init__(self, embed_dim=512):
        super(ImageEncoder, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embed_dim)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input: (batch_size, 3, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x  # (batch_size, embed_dim)


class ResidualBlock(nn.Module):
    """Basic residual block for the encoder"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out


class Attention(nn.Module):
    """Attention mechanism for the decoder"""
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch_size, num_pixels, encoder_dim)
        # decoder_hidden: (batch_size, decoder_dim)
        
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att2 = att2.unsqueeze(1)  # (batch_size, 1, attention_dim)
        
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        
        return attention_weighted_encoding, alpha


class TextDecoder(nn.Module):
    """
    LSTM-based decoder with attention for text generation
    """
    
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        super(TextDecoder, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights"""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
    def init_hidden_state(self, encoder_out):
        """Initialize hidden state from encoder output"""
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation
        
        Args:
            encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
            encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
            caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Sort input data by decreasing lengths
        if caption_lengths.dim() > 1:
            caption_lengths = caption_lengths.squeeze(1)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        decode_lengths = (caption_lengths - 1).tolist()
        
        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)
        
        # At each time-step, decode by attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


class ImageToSceneModel(nn.Module):
    """
    Complete Image-to-Scene Narration Model
    Combines CNN encoder with LSTM decoder and attention
    """
    
    def __init__(self, vocab_size, embed_dim=512, decoder_dim=512, attention_dim=512, dropout=0.5):
        super(ImageToSceneModel, self).__init__()
        
        self.encoder = ImageEncoder(embed_dim)
        self.decoder = TextDecoder(attention_dim, embed_dim, decoder_dim, vocab_size, embed_dim, dropout)
        
    def forward(self, images, captions, caption_lengths):
        """
        Forward propagation
        
        Args:
            images: images, a tensor of dimension (batch_size, 3, 224, 224)
            captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
            caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        """
        # Encode images
        encoder_out = self.encoder(images)  # (batch_size, embed_dim)
        encoder_out = encoder_out.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Decode with attention
        predictions, encoded_captions, decode_lengths, alphas, sort_ind = self.decoder(
            encoder_out, captions, caption_lengths)
        
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    
    def generate_caption(self, image, vocab, max_length=50, repetition_penalty=1.2):
        """
        Generate caption for a single image
        """
        self.eval()
        with torch.no_grad():
            # Encode image (image should already have batch dimension)
            encoder_out = self.encoder(image)  # (1, embed_dim)
            encoder_out = encoder_out.unsqueeze(1)  # (1, 1, embed_dim)
            
            # Initialize decoder
            batch_size = encoder_out.size(0)
            h, c = self.decoder.init_hidden_state(encoder_out)
            
            # Start with <start> token
            start_token = vocab.word2idx['<start>']
            word = torch.tensor([[start_token]]).to(image.device)
            
            generated_caption = []
            alphas = []
            
            for _ in range(max_length):
                # Embed word
                embeddings = self.decoder.embedding(word)  # (1, 1, embed_dim)
                
                # Attention
                attention_weighted_encoding, alpha = self.decoder.attention(encoder_out, h)
                gate = self.decoder.sigmoid(self.decoder.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding
                
                # LSTM step
                h, c = self.decoder.decode_step(
                    torch.cat([embeddings.squeeze(1), attention_weighted_encoding], dim=1),
                    (h, c))
                
                # Predict next word (apply simple repetition controls)
                logits = self.decoder.fc(h)  # (1, vocab_size)
                # Immediate repeat block: avoid repeating the last token
                if len(generated_caption) > 0:
                    last_token = generated_caption[-1]
                    logits[0, last_token] = logits[0, last_token] - 1e9
                # Repetition penalty: down-weight tokens that already appeared
                if len(generated_caption) > 0:
                    with torch.no_grad():
                        unique_tokens = set(generated_caption)
                    for t in unique_tokens:
                        logits[0, t] = logits[0, t] / repetition_penalty
                predicted_id = torch.argmax(logits, dim=1)
                
                # Convert to word
                word = predicted_id.unsqueeze(0)
                generated_caption.append(predicted_id.item())
                alphas.append(alpha.cpu().numpy())
                
                # Stop if <end> token
                if predicted_id.item() == vocab.word2idx['<end>']:
                    break
                    
        return generated_caption, alphas
