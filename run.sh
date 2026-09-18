#!/bin/bash

set -e

INPUT_DIR="${1:-input}"
OUTPUT_DIR="${2:-output}"
SIGMA="${3:-1.5}"

./cuda_image_processor \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --sigma "$SIGMA"
