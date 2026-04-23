from dataclasses import dataclass
import os

# Combined manifest path, relative to repository root (<repo>/clartts/)
DATASET_PATH = 'clartts/combined.parquet'

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# CLArTTS splits under <repo>/clartts/
WAVERNN_TRAIN_PATH = os.path.normpath(
    os.path.join(_REPO_ROOT, 'clartts', 'clartts_train.parquet')
)
WAVERNN_VAL_PATH = os.path.normpath(
    os.path.join(_REPO_ROOT, 'clartts', 'clartts_val.parquet')
)
WAVERNN_TEST_PATH = os.path.normpath(
    os.path.join(_REPO_ROOT, 'clartts', 'clartts_test.parquet')
)

# Shared audio/mel spectrogram parameters
SAMPLE_RATE = 22050
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
FMIN = 0
FMAX = 8000
MIN_DB = -100.0
MAX_SCALED_ABS = 4.0

@dataclass
class Tacotron2Config:
	batch_size: int = 32
	learning_rate: float = 1e-3
	epochs: int = 100
	grad_clip: float = 1.0
	sample_rate: int = SAMPLE_RATE
	n_mels: int = N_MELS
	n_fft: int = N_FFT
	hop_length: int = HOP_LENGTH
	win_length: int = WIN_LENGTH
	fmin: int = FMIN
	fmax: int = FMAX
	min_db: float = MIN_DB
	max_scaled_abs: float = MAX_SCALED_ABS
	seed: int = 42
	checkpoint_dir: str = './checkpoints_taco2/'
	eps : float = 1e-6
	num_workers: int = 0  # DataLoader workers; safe default on Windows; increase on Linux if desired
	### Mel Input Features ###
	num_mels: int = 80 

	num_chars: int = 113 


	### Character Embeddings ###
	character_embed_dim: int = 512
	pad_token_id: int = 0

	### Encoder config ###
	encoder_kernel_size: int = 5
	encoder_n_convolutions: int = 3
	encoder_embed_dim: int = 512
	encoder_dropout_p: float = 0.5

	### Decoder Config ###
	decoder_embed_dim: int = 1024
	decoder_prenet_dim: int = 256
	decoder_prenet_depth: int = 2
	decoder_prenet_dropout_p: float = 0.5
	decoder_postnet_num_convs: int = 5
	decoder_postnet_n_filters: int = 512
	decoder_postnet_kernel_size: int = 5
	decoder_postnet_dropout_p: float = 0.5
	decoder_dropout_p: float = 0.1

	### Attention Config ###
	attention_dim: int = 128
	attention_location_n_filters: int = 32
	attention_location_kernel_size: int = 31
	attention_dropout_p: float = 0.1


@dataclass
class WaveRNNConfig:
	batch_size: int = 8
	learning_rate: float = 1e-4
	weight_decay: float = 1e-6
	epochs: int = 25
	grad_clip: float = 4.0
	num_workers: int = 0
	eval_batches: int = 8
	sample_rate: int = SAMPLE_RATE
	n_mels: int = N_MELS
	n_fft: int = N_FFT
	window_size: int = WIN_LENGTH
	hop_length: int = HOP_LENGTH
	fmin: int = FMIN
	fmax: int = FMAX
	min_db: float = MIN_DB
	max_scaled_abs: float = MAX_SCALED_ABS
	upsample_scales: tuple = (4, 4, 16)
	n_classes: int = 256
	n_res_block: int = 10
	n_rnn: int = 512
	n_fc: int = 512
	kernel_size: int = 5
	n_hidden: int = 128
	n_output: int = 128
	segment_mel_frames: int = 64
	checkpoint_dir: str = './wavernn/checkpoints/'
	checkpoint_name: str = 'wavernn_last.pt'
	# Per-epoch loss plots + sample WAVs (relative to repo cwd)
	monitor_dir: str = './wavernn/training_monitor/'
	seed: int = 42
	resume_from: str = None



