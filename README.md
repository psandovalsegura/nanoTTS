# nanoTTS

**Goal**: build a minimal, simple, hackable text-to-speech (TTS) system using a GPT-2-style transformer decoder.

**Key design choices**:
- Look like an LLM: tokens in, tokens out. No extra modules, no added complexity.
- Stay as simple as possible. Follow [nanoGPT](https://github.com/karpathy/nanoGPT)