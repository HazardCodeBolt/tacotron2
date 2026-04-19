import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from commons.dataset import TTSDataset


def main():
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parquet_path = os.path.normpath(
        os.path.join(_repo_root, "clartts", "clartts_test.parquet")
    )
    dataset = TTSDataset(parquet_path)
    print(f"Loaded {len(dataset)} samples from {parquet_path}")
    transcript, mel = dataset[0]
    print('Transcript:', transcript)
    print('Mel shape:', mel.shape)
    assert isinstance(transcript, str)
    assert isinstance(mel, torch.Tensor)
    assert mel.ndim == 2

if __name__ == "__main__":
    main()
