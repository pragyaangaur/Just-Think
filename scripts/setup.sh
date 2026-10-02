#!/bin/sh
# Fetch everything that is not stored in this repository: the Pain Axis release (for the
# vectors and sentence sets) at the commit this study used, and the model weights.
set -e
cd "$(dirname "$0")/.."

if [ ! -d external/Pain-axis ]; then
  git clone https://github.com/valen-research/Pain-axis external/Pain-axis
fi
git -C external/Pain-axis checkout 4d75cd90e206ea962f7a9101e65c85efea56723b

# Main-study weights (Apple silicon only). About 4.3 GB. curl resumes if the download stalls.
M=models/Qwen2.5-7B-Instruct-4bit
REV=c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed
mkdir -p "$M"
for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json model.safetensors; do
  until curl -sSL -C - --speed-limit 100000 --speed-time 30 -o "$M/$f" \
      "https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit/resolve/$REV/$f"; do
    echo "retrying $f"
  done
done
echo "setup done"
