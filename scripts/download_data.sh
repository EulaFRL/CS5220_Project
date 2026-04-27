#!/usr/bin/env bash
# Download Fashion-MNIST IDX binary files into data/fashion-mnist/
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/fashion-mnist"
mkdir -p "$DEST"

# S3 HTTP mirror (sometimes blocked on HPC login); GitHub raw HTTPS as fallback.
URLS_S3="http://fashion-mnist.s3-website.eu-west-1.amazonaws.com"
URLS_GH="https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

fetch_one() {
    local f="$1" gz="$DEST/$f" url="$2"
    rm -f "$gz"
    wget --tries=5 --timeout=60 --read-timeout=60 -O "$gz" "${url}/$f"
}

for f in "${FILES[@]}"; do
    out="${DEST}/${f%.gz}"
    if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        continue
    fi
    echo "[download] $f"
    gz="${DEST}/$f"
    rm -f "$gz"
    if ! fetch_one "$f" "$URLS_S3" 2>/dev/null; then
        echo "[retry] $f via GitHub (HTTPS)"
        fetch_one "$f" "$URLS_GH"
    fi
    if ! gzip -t "$gz" 2>/dev/null; then
        echo "[retry] $f corrupt or not gzip — trying GitHub"
        rm -f "$gz"
        fetch_one "$f" "$URLS_GH"
        gzip -t "$gz"
    fi
    gunzip "$gz"
done

echo "[done] Fashion-MNIST data in $DEST"
ls -lh "$DEST"
