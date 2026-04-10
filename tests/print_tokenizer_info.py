
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tacotron2.tokenizer import Tokenizer

tokenizer = Tokenizer()
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")
print(f"Tokenizer unk_token_id: {tokenizer.unk_token_id}")
