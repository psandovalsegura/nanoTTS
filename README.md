# nanoTTS

**goal**: build a minimal, simple, hackable text-to-speech (TTS) system using a GPT-2-style transformer decoder.

**key design decisions**:
1. look like an LLM: tokens in, tokens out. no extra modules, no added complexity.
2. keep it as simple as possible. follow [nanoGPT](https://github.com/karpathy/nanoGPT).

## install

1. install [WavTokenizer](https://github.com/jishengpeng/WavTokenizer?tab=readme-ov-file)
2. additional packages in requirements.txt

## other

- to retrain BPE tokenizer: `python libritts_tokenizer.py` which will use transcripts and ids from **libritts_tokenizer_data.pkl** and save **libritts_bpe.json**.