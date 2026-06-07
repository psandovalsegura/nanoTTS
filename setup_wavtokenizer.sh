#!/usr/bin/env bash
set -euo pipefail

echo "=> Step 1 of 4: Cloning WavTokenizer repo"
if [ -d ./WavTokenizer ]; then
  echo "   Already exists, skipping."
else
  git clone https://github.com/jishengpeng/WavTokenizer ./WavTokenizer
fi

echo "=> Step 2 of 4: Writing pyproject.toml"
if [ -f ./WavTokenizer/pyproject.toml ]; then
  echo "   Already exists, skipping."
else
cat > ./WavTokenizer/pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "wavtokenizer"
version = "0.0.1"

[tool.setuptools]
packages = ["encoder", "decoder"]
EOF
fi

echo "=> Step 3 of 4: Installing wavtokenizer package"
# @psando: TODO these requirements from WavTokenizer aren't needed for 
#          training, but I think they were needed for preproprecess_audio.py?
# pip install -r ./WavTokenizer/requirements.txt
pip install -e ./WavTokenizer

echo "=> Step 4 of 4: Downloading model weights from HuggingFace"
if [ -d ./hf_hub/WavTokenizer ]; then
  echo "   Already exists, skipping."
else
  mkdir -p ./hf_hub
  hf download novateur/WavTokenizer --local-dir ./hf_hub/WavTokenizer
fi

echo ""
echo "Done. Model files in /workspace/hf_hub/WavTokenizer/"
