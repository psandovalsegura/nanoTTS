"""
Dataset for TTS training over pre-encoded WavTokenizer audio codes on disk.

Cache layout produced by preprocess_audio.py:
    {cache_dir}/audio_codes.bin   uint16, concatenated raw codes (0..4095)
    {cache_dir}/index.npy         int64 (N+1,), cumulative offsets
    {cache_dir}/texts.jsonl       {"utterance_id", "text"} per line
    {cache_dir}/meta.json         provenance / config

`__getitem__` returns `(X, Y)` tensors ready for next-token-prediction:
text positions in Y are masked to -100, and audio targets are remapped
into [0, audio_vocab_size). Pair with `create_collate_fn` in the DataLoader.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class PretokenizedTTSDataset(Dataset):
    def __init__(self, cache_dir, tokenizer, max_seq_len=None):
        cache_dir = Path(cache_dir)
        assert cache_dir.is_dir(), f"cache dir not found: {cache_dir}"

        meta = json.loads((cache_dir / "meta.json").read_text())
        assert meta["audio_vocab_size"] == tokenizer.audio_vocab_size, (
            f"cache audio_vocab_size {meta['audio_vocab_size']} != "
            f"tokenizer.audio_vocab_size {tokenizer.audio_vocab_size}"
        )
        assert meta["dtype"] == "uint16"

        self.cache_dir = cache_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.pad_id = tokenizer.pad_id
        self.astart_id = tokenizer.audio_start_id
        self.in_eos_id = tokenizer.in_eos_id
        self.out_eos_id = tokenizer.out_eos_id
        self.audio_offset = tokenizer.audio_offset

        self.index = np.load(cache_dir / "index.npy")
        with open(cache_dir / "texts.jsonl") as f:
            entries = [json.loads(line) for line in f]
        self.texts = [e["text"] for e in entries]

        assert len(self.index) == len(self.texts) + 1, (
            f"index ({len(self.index)}) and texts ({len(self.texts)}) disagree"
        )
        # memmap is per-instance; in DataLoader workers each gets its own
        self._codes_path = cache_dir / "audio_codes.bin"
        self._codes = None  # lazy-open in workers

    def __len__(self):
        return len(self.texts)

    @property
    def codes(self):
        if self._codes is None:
            self._codes = np.memmap(self._codes_path, dtype=np.uint16, mode="r")
        return self._codes

    def __getitem__(self, idx):
        s, e = int(self.index[idx]), int(self.index[idx + 1])
        raw_codes = self.codes[s:e].astype(np.int64)
        audio_ids = (raw_codes + self.audio_offset).tolist()

        text_ids = self.tokenizer.encode_text(f"{self.tokenizer.BOS_TOKEN}{self.texts[idx]}{self.tokenizer.AUDIO_START_TOKEN}")

        sequence = text_ids + audio_ids + [self.in_eos_id]
        if self.max_seq_len is not None and len(sequence) > self.max_seq_len:
            sequence = sequence[:self.max_seq_len]
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)

        X = sequence_tensor[:-1]
        Y = sequence_tensor[1:].clone()

        astart_idx = (Y == self.astart_id).nonzero(as_tuple=True)[0].item()
        Y[:astart_idx + 1] = -100 # @psando: mask text and audio start token in loss

        Y[Y >= self.audio_offset] -= self.audio_offset
        Y[Y == self.in_eos_id] = self.out_eos_id
        return X, Y


def create_collate_fn(pad_id):
    def collate_fn(batch):
        xs, ys = zip(*batch)
        # pad X with the BPE's <PAD> id
        # pad Y with -100 so padding doesn't affect the loss
        x_padded = pad_sequence(xs, batch_first=True, padding_value=pad_id)
        y_padded = pad_sequence(ys, batch_first=True, padding_value=-100)
        return x_padded, y_padded
    return collate_fn
