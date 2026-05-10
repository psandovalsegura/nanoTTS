# nanoTTS

**goal**: a minimal, simple, hackable text-to-speech (TTS) system using a GPT-2-style transformer decoder.

![nanoTTS](assets/nanoTTS_overview.jpeg)

**demo**: [here's a sample](https://x.com/psandovalsegura/status/2040905729545220167?s=20) generated after ~9 epochs over 53.78 hours of English text + audio pairs.

---

**Update May 2026** Removed speaker IDs, and thus removed the ability to choose a voice at inference time, because this choice prevented us from training on any dataset other than the train-clean-100 split of LibriTTS. This was a huge limitation. Now that the input sequence encoding is even simpler, we can train on any dataset of text-audio pairs and monitor a val loss! I updated the train script to use both clean train splits of LibriTTS (which total ~245 hours).

---


**key design decisions**:
1. look like an LLM: tokens in, tokens out. no extra modules, no added complexity.
2. keep architecture as simple as possible. follow [nanoGPT](https://github.com/karpathy/nanoGPT).
3. audio tokens from [WavTokenizer](https://github.com/jishengpeng/WavTokenizer?tab=readme-ov-file): a SOTA approach for converting audio to a short sequence of discrete tokens

## install

1. install [WavTokenizer](https://github.com/jishengpeng/WavTokenizer?tab=readme-ov-file). you will need to add a *pyproject.toml* file to your clone of that repo so that you can install `packages = ["encoder", "decoder"]`.
2. additional packages in requirements.txt.

## table of contents

| file | purpose |
| --- | --- |
| `train.py` | main training script; loads LibriTTS, builds the model/tokenizers, and runs training or resume-from-checkpoint |
| `model.py` | GPT-style decoder-only transformer used for next-token prediction over the joint text/audio sequence |
| `libritts_dataset.py` | dataset wrapper that converts raw LibriTTS examples into `[BOS, text tokens, AUDIO_START, audio tokens, EOS]` training sequences |
| `tokenizer.py` | joint tokenizer interface that combines the text tokenizer with WavTokenizer audio codes and handles decode for inference |
| `configurator.py` | lightweight config override helper used by `train.py` for command-line and file-based hyperparameter overrides |
| `text_tokenizer/libritts_tokenizer.py` | script that trains the BPE text tokenizer |
| `text_tokenizer/libritts_bpe.json` | serialized BPE tokenizer used for transcript text plus special tokens |
| `text_tokenizer/train-clean-100.json` | saved transcripts to build the text tokenizer |

## other
- to retrain BPE text tokenizer: in *text_tokenizer/* run `python libritts_tokenizer.py` which will use transcripts from *train-clean-100.json* and save *libritts_bpe.json*.

## cite

```bibtex
@misc{sandovalsegura2026nanotts,
  title={nanoTTS: Minimal Text-to-Speech using nanoGPT},
  author={Pedro Sandoval-Segura},
  year={2026},
  note={GitHub repository}
}
```
