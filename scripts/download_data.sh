#!/usr/bin/env bash
# Download Fashion-MNIST IDX binary files into data/fashion-mnist/
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/fashion-mnist"
mkdir -p "$DEST"

BASE="http://fashion-mnist.s3-website.eu-west-1.amazonaws.com"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

for f in "${FILES[@]}"; do
    out="${DEST}/${f%.gz}"
    if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        continue
    fi
    echo "[download] $f"
    wget -q "$BASE/$f" -O "$DEST/$f"
    gunzip "$DEST/$f"
done

echo "[done] Fashion-MNIST data in $DEST"
ls -lh "$DEST"
