"""
Dataset builders for preprocess_audio.py.

A "builder" is a callable that returns a `torch.utils.data.Dataset`
whose `__getitem__(idx)` yields a `Sample(utterance_id, text, waveform)`.
The waveform must be MONO float32 at 24 kHz. Resampling is the
builder's job, not the encoder's.

To add a new dataset:
  1. Write a Dataset subclass that returns `Sample` items.
  2. Write `def my_dataset(root, **kwargs) -> Dataset` that constructs it.
  3. Register: BUILDERS["my_dataset"] = my_dataset.
"""

from typing import Callable, NamedTuple

import torch
import torchaudio
from torch.utils.data import Dataset


class Sample(NamedTuple):
    utterance_id: str
    text: str
    waveform: torch.Tensor  # 1-D float32, mono, 24 kHz


_EXPECTED_SR = 24000


class _LibriTTSBuilder(Dataset):
    def __init__(self, root: str, url: str):
        self.dataset = torchaudio.datasets.LIBRITTS(root=root, url=url, download=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Sample:
        waveform, sr, _orig_text, normalized_text, speaker_id, chapter_id, utt_id = self.dataset[idx]
        assert sr == _EXPECTED_SR, f"LibriTTS sample {utt_id} has sr={sr}, expected {_EXPECTED_SR}"
        wav = waveform.squeeze(0).to(torch.float32)
        assert wav.ndim == 1, f"expected mono waveform, got shape {tuple(waveform.shape)}"
        uid = f"{speaker_id}_{chapter_id}_{utt_id}"
        return Sample(utterance_id=uid, text=normalized_text, waveform=wav)


def libritts(root: str, url: str) -> Dataset:
    return _LibriTTSBuilder(root=root, url=url)


BUILDERS: dict[str, Callable[..., Dataset]] = {
    "libritts": libritts,
}
