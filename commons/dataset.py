import random

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset
import librosa
from tacotron2.tokenizer import Tokenizer

import numpy as np

def load_wav(path_to_audio, sr=22050):
    audio, orig_sr = torchaudio.load(path_to_audio)

    if sr != orig_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=sr)

    return audio.squeeze(0)

def amp_to_db(x, min_db=-100):
    ### Forces min DB to be -100
    ### 20 * torch.log10(1e-5) = 20 * -5 = -100
    clip_val = 10 ** (min_db / 20)
    return 20 * torch.log10(torch.clamp(x, min=clip_val))

def db_to_amp(x):
    return 10 ** (x / 20)

def normalize(x, 
              min_db=-100., 
              max_abs_val=4):

    x = (x - min_db) / -min_db
    x = 2 * max_abs_val * x - max_abs_val
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    
    return x

def denormalize(x, 
                min_db=-100, 
                max_abs_val=4):
    
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    x = (x + max_abs_val) / (2 * max_abs_val)
    x = x * -min_db + min_db

    return x

class AudioMelConversions:
    def __init__(self,
                 num_mels=80,
                 sampling_rate=22050, 
                 n_fft=1024, 
                 window_size=1024, 
                 hop_size=256,
                 fmin=0, 
                 fmax=8000,
                 center=False,
                 min_db=-100, 
                 max_scaled_abs=4):
        
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        self.spec2mel = self._get_spec2mel_proj()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)

    def _get_spec2mel_proj(self):
        mel = librosa.filters.mel(sr=self.sampling_rate, 
                                  n_fft=self.n_fft, 
                                  n_mels=self.num_mels, 
                                  fmin=self.fmin, 
                                  fmax=self.fmax)
        return torch.from_numpy(mel)
    
    def audio2mel(self, audio, do_norm=False):

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        spectrogram = torch.stft(input=audio, 
                                 n_fft=self.n_fft, 
                                 hop_length=self.hop_size, 
                                 win_length=self.window_size, 
                                 window=torch.hann_window(self.window_size).to(audio.device), 
                                 center=self.center, 
                                 pad_mode="reflect", 
                                 normalized=False, 
                                 onesided=True,
                                 return_complex=True)
        
        spectrogram = torch.abs(spectrogram)
        
        mel = torch.matmul(self.spec2mel.to(spectrogram.device), spectrogram)

        mel = amp_to_db(mel, self.min_db)
        
        if do_norm:
            mel = normalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        return mel
    
    def mel2audio(self, mel, do_denorm=False, griffin_lim_iters=60):

        if do_denorm:
            mel = denormalize(mel, min_db=self.min_db, max_abs_val=self.max_scaled_abs)

        mel = db_to_amp(mel)

        device = mel.device
        dtype = mel.dtype
        spectrogram = torch.matmul(self.mel2spec.to(device=device, dtype=dtype), mel)

        window = torch.hann_window(self.window_size, device=device, dtype=dtype)
        # Magnitude spectrogram + Griffin–Lim via torchaudio (aligned with torch.stft in audio2mel).
        audio = torchaudio.functional.griffinlim(
            spectrogram,
            window,
            self.n_fft,
            self.hop_size,
            self.window_size,
            power=1.0,
            n_iter=griffin_lim_iters,
            momentum=0.99,
            length=None,
            rand_init=True,
        )

        audio = audio.reshape(-1).float()
        peak = torch.amax(torch.abs(audio)).clamp(min=0.01)
        audio_i16 = (audio * (32767.0 / peak)).round().to(torch.int16).cpu().numpy()

        return audio_i16

def build_padding_mask(lengths):

    B = lengths.size(0)
    T = torch.max(lengths).item()

    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, lengths[i]:] = 1

    return mask.bool()

class TTSDataset(Dataset):
    def __init__(self, 
                 path_to_metadata,
                 sample_rate=22050,
                 n_fft=1024, 
                 window_size=1024, 
                 hop_size=256, 
                 fmin=0,
                 fmax=8000, 
                 num_mels=80, 
                 center=False, 
                 normalized=False, 
                 min_db=-100, 
                 max_scaled_abs=4):
        
        # Support both CSV and Parquet for metadata
        self.is_parquet = path_to_metadata.endswith('.parquet')
        if self.is_parquet:
            self.metadata = pd.read_parquet(path_to_metadata)
        else:
            self.metadata = pd.read_csv(path_to_metadata)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax 
        self.num_mels = num_mels
        self.center = center
        self.normalized = normalized
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs

        if self.is_parquet:
            self.transcript_lengths = [len(Tokenizer().encode(t)) for t in self.metadata["text"]]
        else:
            self.transcript_lengths = [len(Tokenizer().encode(t)) for t in self.metadata["normalized_transcript"]]

        self.audio_proc = AudioMelConversions(num_mels=self.num_mels, 
                                              sampling_rate=self.sample_rate, 
                                              n_fft=self.n_fft, 
                                              window_size=self.win_size, 
                                              hop_size=self.hop_size, 
                                              fmin=self.fmin, 
                                              fmax=self.fmax, 
                                              center=self.center,
                                              min_db=self.min_db, 
                                              max_scaled_abs=self.max_scaled_abs)
        
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        if self.is_parquet:
            transcript = sample["text"]
            # Load audio directly from the 'audio' column (expects a numpy array or list)
            audio = sample["audio"]
            if isinstance(audio, (np.ndarray, list)):
                audio = torch.tensor(audio, dtype=torch.float32)
            else:
                # If stored as bytes, decode to numpy array (assume float32 PCM)
                audio = torch.from_numpy(np.frombuffer(audio, dtype=np.float32))
            # Optionally resample if needed
            if hasattr(sample, "sr") and sample["sr"] != self.sample_rate:
                audio = torchaudio.functional.resample(audio, orig_freq=sample["sr"], new_freq=self.sample_rate)
        else:
            transcript = sample["normalized_transcript"]
            path_to_audio = sample["file_path"]
            audio = load_wav(path_to_audio, sr=self.sample_rate)

        mel = self.audio_proc.audio2mel(audio, do_norm=True)
        return transcript, mel.squeeze(0)

def TTSCollator():

    tokenizer = Tokenizer()

    def _collate_fn(batch):
        
        texts = [tokenizer.encode(b[0]) for b in batch]
        mels = [b[1] for b in batch]
        
        ### Get Lengths of Texts and Mels ###
        input_lengths = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
        output_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)

        ### Sort by Text Length (as we will be using packed tensors later) ###
        input_lengths, sorted_idx = input_lengths.sort(descending=True)
        texts = [texts[i] for i in sorted_idx]
        mels = [mels[i] for i in sorted_idx]
        output_lengths = output_lengths[sorted_idx]

        ### Pad Text ###
        text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)

        ### Pad Mel Sequences ###
        max_target_len = max(output_lengths).item()
        num_mels = mels[0].shape[0]
        
        ### Get gate which tells when to stop decoding. 0 is keep decoding, 1 is stop ###
        mel_padded = torch.zeros((len(mels), num_mels, max_target_len))
        gate_padded = torch.zeros((len(mels), max_target_len))

        for i, mel in enumerate(mels):
            t = mel.shape[1]
            mel_padded[i, :, :t] = mel
            gate_padded[i, t-1:] = 1
        
        mel_padded = mel_padded.transpose(1,2)

        return text_padded, input_lengths, mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths)


    return _collate_fn

class BatchSampler:
    def __init__(self, dataset, batch_size, drop_last=False):
        self.sampler = torch.utils.data.SequentialSampler(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_batches = self._make_batches()

    def _make_batches(self):

        indices = [i for i in self.sampler]

        if self.drop_last:

            total_size = (len(indices) // self.batch_size) * self.batch_size
            indices = indices[:total_size]

        batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        random_indices = torch.randperm(len(batches))
        return [batches[i] for i in random_indices]
    
    def __iter__(self):
        for batch in self.random_batches:
            yield batch

    def __len__(self):
        return len(self.random_batches)


class WaveRNNDataset(Dataset):
    """Random mel/wave segments for WaveRNN training, aligned with :class:`TTSDataset` preprocessing.

    Uses the same parquet/CSV loading and :class:`AudioMelConversions` settings as Tacotron2 so
    mel statistics match. Expects a config object with the same fields as ``WaveRNNConfig`` in
    ``commons.hyperparams`` (``sample_rate``, ``n_mels``, ``n_fft``, ``hop_length``, ``window_size``,
    ``fmin``, ``fmax``, ``min_db``, ``max_scaled_abs``, ``segment_mel_frames``, ``kernel_size``,
    ``n_classes``).

    Each item is ``(mel_seg, waveform_in, target)`` where ``mel_seg`` has
    ``segment_mel_frames + kernel_size - 1`` frames (e.g. ``segment_mel_frames + 4`` when
    ``kernel_size == 5``), ``waveform_in`` is float audio ``[-1, 1]`` of length
    ``segment_mel_frames * hop_length``, and ``target`` is mu-law indices for cross-entropy.
    """

    def __init__(self, path_to_metadata, config):
        self.is_parquet = path_to_metadata.endswith('.parquet')
        if self.is_parquet:
            self.metadata = pd.read_parquet(path_to_metadata)
        else:
            self.metadata = pd.read_csv(path_to_metadata)

        self.config = config
        self.sample_rate = config.sample_rate
        self.n_fft = config.n_fft
        self.win_size = config.window_size
        self.hop_size = config.hop_length
        self.fmin = config.fmin
        self.fmax = config.fmax
        self.num_mels = config.n_mels
        self.min_db = config.min_db
        self.max_scaled_abs = config.max_scaled_abs

        self.kernel_size = config.kernel_size
        self.segment_mel_frames = config.segment_mel_frames
        self.n_classes = config.n_classes

        self.t_mel = self.segment_mel_frames + self.kernel_size - 1
        self.n_audio = self.segment_mel_frames * self.hop_size

        self.audio_proc = AudioMelConversions(
            num_mels=self.num_mels,
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            window_size=self.win_size,
            hop_size=self.hop_size,
            fmin=self.fmin,
            fmax=self.fmax,
            center=False,
            min_db=self.min_db,
            max_scaled_abs=self.max_scaled_abs,
        )

    def __len__(self):
        return len(self.metadata)

    def _row_audio(self, sample):
        if self.is_parquet:
            audio = sample['audio']
            if isinstance(audio, (np.ndarray, list)):
                audio = torch.tensor(audio, dtype=torch.float32)
            else:
                audio = torch.from_numpy(np.frombuffer(audio, dtype=np.float32))
            if 'sr' in sample.index and sample['sr'] != self.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, orig_freq=int(sample['sr']), new_freq=self.sample_rate
                )
        else:
            path_to_audio = sample['file_path']
            audio = load_wav(path_to_audio, sr=self.sample_rate)
        return audio.reshape(-1)

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        audio = self._row_audio(sample)
        mel = self.audio_proc.audio2mel(audio, do_norm=True)

        t = mel.shape[1]
        if t < self.t_mel:
            mel = F.pad(mel, (0, self.t_mel - t))
            max_start = 0
        else:
            max_start = t - self.t_mel

        start = random.randint(0, max_start)
        mel_seg = mel[:, start : start + self.t_mel].contiguous()

        a0 = start * self.hop_size
        wave_seg = audio[a0 : a0 + self.n_audio]
        if wave_seg.numel() < self.n_audio:
            wave_seg = F.pad(wave_seg, (0, self.n_audio - wave_seg.numel()))

        wave_seg = torch.clamp(wave_seg, -1.0, 1.0)
        target = AF.mu_law_encoding(wave_seg, quantization_channels=self.n_classes)

        return mel_seg, wave_seg, target
