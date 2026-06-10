"""ECG-Byte: quantize a normalized signal to symbols, BPE-encode via the Rust
extension (data/bpe). The trained vocab/merges ship as a pickle."""
import pickle
import string

import numpy as np

SYMBOLS = np.array(list(string.ascii_lowercase))


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)  # (vocab, merges)


def ecg_vocab_tokens(path):
    from registry import ECG_TOKEN_PREFIX
    vocab, _ = _load(path)
    return [f"{ECG_TOKEN_PREFIX}{token_id}" for token_id in vocab]


class EcgByte:
    def __init__(self, path):
        import bpe  # compiled from data/bpe (maturin)
        self._encode = bpe.encode_symbol
        self.vocab, self.merges = _load(path)

    def encode(self, normalized_signal):
        quantized = np.minimum((normalized_signal * len(SYMBOLS)).astype(np.uint8), len(SYMBOLS) - 1)
        symbols = "".join(SYMBOLS[quantized].flatten())
        return self._encode(symbols, self.merges)
