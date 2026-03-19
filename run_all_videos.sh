#!/bin/bash

for video in ./videos/*.mp4; do
    filename=$(basename "$video")
    echo "=== Processing: $filename ==="
    python3 -m src.main --video "$filename" -v --headless
    echo "=== Done: $filename ==="
    echo
done
