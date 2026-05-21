"""
Pre-encode a text-audio dataset with WavTokenizer and cache discrete codes
on disk for fast training.

Each utterance is encoded individually (batch_size=1 on the GPU) so codes
match what `JointTokenizer.encode_audio` would have produced on-the-fly. The
encoder is a non-causal SEANet with reflect padding; batched zero-padded
inputs perturb codes near each utterance's tail.

Layout written to {output_dir}/{dataset}/{audio_tokenizer_tag}/{split}/:
  audio_codes.bin   uint16 LE, concatenated raw WavTokenizer codes (0..4095)
  index.npy         int64 (N+1,), cumulative code offsets
  texts.jsonl       {"utterance_id", "text"} per line, line i == utterance i
  meta.json         provenance / configuration
  skipped.jsonl     utterances dropped (too short / NaN / etc.)
  truncated.jsonl   utterances truncated to --max-audio-seconds
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from math import prod
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from decoder.pretrained import WavTokenizer
from dataset_builders import BUILDERS, Sample


PREPROCESS_AUDIO_VERSION = "2"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(BUILDERS.keys()))
    p.add_argument("--split", required=True, help="builder-specific split name (e.g. train-clean-100)")
    p.add_argument("--dataset-root", required=True, help="root passed to the builder")
    p.add_argument("--output-dir", required=True, help="cache root")
    p.add_argument("--wavtokenizer-dir", required=True)
    p.add_argument("--wavtokenizer-config", required=True, help="YAML filename inside --wavtokenizer-dir")
    p.add_argument("--wavtokenizer-ckpt", required=True, help="checkpoint filename inside --wavtokenizer-dir")
    p.add_argument("--audio-tokenizer-tag", default=None,
                   help="subdir name; defaults to wavtokenizer_{frame_rate}tok")
    p.add_argument("--batch-size", type=int, default=16,
                   help="DataLoader batch size (I/O parallelism only; GPU encode is per-item)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-loader-workers", type=int, default=4)
    p.add_argument("--max-audio-seconds", type=float, default=50.0,
                   help="truncate audio longer than this; pass 0 to disable")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hop_length_and_sr(config_path: Path) -> tuple[int, int]:
    cfg = yaml.safe_load(config_path.read_text())
    init = cfg["model"]["init_args"]
    sample_rate = int(init["sample_rate"])
    dowmsamples = init["feature_extractor"]["init_args"]["dowmsamples"]
    return prod(dowmsamples), sample_rate


def collate(samples: list[Sample]):
    ids = [s.utterance_id for s in samples]
    texts = [s.text for s in samples]
    wavs = [s.waveform for s in samples]
    lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    T_max = int(lengths.max().item())
    padded = torch.zeros(len(samples), T_max, dtype=torch.float32)
    for j, w in enumerate(wavs):
        padded[j, : w.shape[0]] = w
    return ids, texts, padded, lengths


def main():
    args = parse_args()

    config_path = Path(args.wavtokenizer_dir) / args.wavtokenizer_config
    ckpt_path = Path(args.wavtokenizer_dir) / args.wavtokenizer_ckpt
    assert config_path.exists(), f"config not found: {config_path}"
    assert ckpt_path.exists(), f"ckpt not found: {ckpt_path}"

    hop_length, sample_rate = parse_hop_length_and_sr(config_path)
    frame_rate = sample_rate // hop_length
    tag = args.audio_tokenizer_tag or f"wavtokenizer_{frame_rate}tok"
    out_dir = Path(args.output_dir) / args.dataset / tag / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"hop_length={hop_length}, sample_rate={sample_rate}, frame_rate={frame_rate}")
    print(f"writing to: {out_dir}")

    # always start fresh; clear any stale outputs from prior runs
    for fname in ("audio_codes.bin", "index.npy", "texts.jsonl", "meta.json",
                   "skipped.jsonl", "truncated.jsonl"):
        p = out_dir / fname
        if p.exists():
            p.unlink()

    print(f"loading WavTokenizer on {args.device}...")
    wavtok = WavTokenizer.from_pretrained0802(str(config_path), str(ckpt_path))
    wavtok = wavtok.to(args.device).eval()
    bandwidth_id = torch.tensor([0], device=args.device)

    full_ds = BUILDERS[args.dataset](root=args.dataset_root, url=args.split)
    total_n = len(full_ds)
    loader = DataLoader(
        full_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_loader_workers,
        collate_fn=collate,
        pin_memory=(args.device != "cpu"),
        persistent_workers=(args.num_loader_workers > 0),
    )

    max_samples = int(args.max_audio_seconds * sample_rate) if args.max_audio_seconds > 0 else None

    index = [0]
    num_skipped = 0
    num_truncated = 0
    pbar = tqdm(total=total_n, unit="utt", smoothing=0.05)

    with open(out_dir / "audio_codes.bin", "wb") as bin_fp, \
         open(out_dir / "texts.jsonl", "w") as txt_fp, \
         open(out_dir / "skipped.jsonl", "w") as skipped_fp, \
         open(out_dir / "truncated.jsonl", "w") as truncated_fp:

        for ids, texts, padded, lengths in loader:
            for j in range(len(ids)):
                L = int(lengths[j].item())
                wav_j = padded[j, :L]
                if L < hop_length or not torch.isfinite(wav_j).all():
                    reason = "too_short" if L < hop_length else "non_finite"
                    skipped_fp.write(json.dumps({
                        "utterance_id": ids[j], "reason": reason, "length_samples": L,
                    }) + "\n")
                    num_skipped += 1
                    pbar.update(1)
                    continue
                if max_samples is not None and L > max_samples:
                    truncated_fp.write(json.dumps({
                        "utterance_id": ids[j],
                        "original_samples": L,
                        "truncated_to": max_samples,
                    }) + "\n")
                    L = max_samples
                    wav_j = wav_j[:L]
                    num_truncated += 1

                wav_in = wav_j.to(args.device, non_blocking=True).unsqueeze(0)  # (1, L)
                with torch.inference_mode():
                    _, codes = wavtok.encode_infer(wav_in, bandwidth_id=bandwidth_id)
                # codes shape: (n_q=1, B=1, T_codes_j), int64 in [0, 4095]
                assert codes.dim() == 3 and codes.shape[0] == 1 and codes.shape[1] == 1, \
                    f"unexpected codes shape: {tuple(codes.shape)}"
                assert int(codes.max().item()) < 4096, "code >= 4096; wrong config/ckpt?"
                code_row = codes[0, 0].to(torch.int32).cpu().numpy().astype(np.uint16, copy=False)

                bin_fp.write(code_row.tobytes())
                index.append(index[-1] + int(code_row.shape[0]))
                txt_fp.write(json.dumps({"utterance_id": ids[j], "text": texts[j]}) + "\n")
                pbar.update(1)

    pbar.close()

    index_arr = np.array(index, dtype=np.int64)
    np.save(out_dir / "index.npy", index_arr)

    num_utterances = len(index_arr) - 1
    meta = {
        "dataset_name": args.dataset,
        "split": args.split,
        "wavtokenizer_config": args.wavtokenizer_config,
        "wavtokenizer_ckpt": args.wavtokenizer_ckpt,
        "wavtokenizer_config_sha256": sha256_file(config_path),
        "wavtokenizer_ckpt_sha256": sha256_file(ckpt_path),
        "audio_tokenizer_tag": tag,
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
        "hop_length": hop_length,
        "audio_vocab_size": 4096,
        "dtype": "uint16",
        "endianness": "little",
        "num_utterances": num_utterances,
        "total_audio_tokens": int(index_arr[-1]),
        "num_skipped": num_skipped,
        "num_truncated": num_truncated,
        "preprocess_audio_version": PREPROCESS_AUDIO_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"done: {num_utterances:,} utterances, {int(index_arr[-1]):,} audio tokens, "
          f"{num_skipped} skipped, {num_truncated} truncated")
    print(f"audio_codes.bin: {(out_dir / 'audio_codes.bin').stat().st_size / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
