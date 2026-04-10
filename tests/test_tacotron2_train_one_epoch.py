
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from commons.hyperparams import Tacotron2Config, DATASET_PATH
from commons.dataset import TTSDataset, TTSCollator
from tacotron2.model import Tacotron2
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Load config and dataset
config = Tacotron2Config()
dataset = TTSDataset(DATASET_PATH)
collate_fn = TTSCollator()
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Model
model = Tacotron2(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Single epoch train loop
model.train()
for batch in loader:
    text_padded, input_lengths, mel_padded, gate_padded, encoder_mask, decoder_mask = batch
    optimizer.zero_grad()
    mel_outs, mel_postnet_out, stop_tokens, attention_weights = model(
        text_padded, input_lengths, mel_padded, encoder_mask, decoder_mask
    )
    mel_loss = F.mse_loss(mel_outs, mel_padded)
    refined_mel_loss = F.mse_loss(mel_postnet_out, mel_padded)
    stop_loss = F.binary_cross_entropy_with_logits(stop_tokens.reshape(-1,1), gate_padded.reshape(-1,1))
    loss = mel_loss + refined_mel_loss + stop_loss
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f} | Mel: {mel_loss.item():.4f} | RMel: {refined_mel_loss.item():.4f} | Stop: {stop_loss.item():.4f}")
    break  # Only one batch for quick test
