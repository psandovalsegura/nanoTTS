# nanoTTS

**goal**: a minimal, simple, hackable text-to-speech (TTS) system using a GPT-2-style transformer decoder.

**key design decisions**:
1. look like an LLM: tokens in, tokens out. no extra modules, no added complexity.
2. keep architecture as simple as possible. follow [nanoGPT](https://github.com/karpathy/nanoGPT).
3. audio tokens from wavtokenizer: a SOTA approach for converting audio to a short sequence of discrete tokens

## install

1. install [WavTokenizer](https://github.com/jishengpeng/WavTokenizer?tab=readme-ov-file). you will need to add a *pyproject.toml* file to your clone of that repo so that you can install `packages = ["encoder", "decoder"]`.
2. additional packages in requirements.txt.

## other

- to retrain BPE tokenizer: `python libritts_tokenizer.py` which will use transcripts and ids from *libritts_tokenizer_data.pkl* and save *libritts_bpe.json*.