# CUDA GPU Image Processing Pipeline

## Project Overview

This project demonstrates GPU-accelerated batch image processing using
custom NVIDIA CUDA kernels.

The program processes 120 RGB images and performs two image-processing
operations on the GPU:

1. RGB-to-grayscale conversion.
2. Gaussian blur.

The project was developed from scratch using CUDA C++ and performs the
core image-processing operations on the NVIDIA GPU.

## GPU Computation

The project contains two CUDA kernels.

### Grayscale Kernel

Each CUDA thread processes one image pixel.

The RGB channels are converted to grayscale using:

gray = 0.299R + 0.587G + 0.114B

This allows many pixels to be processed concurrently by the GPU.

### Gaussian Blur Kernel

The second CUDA kernel applies a 5x5 Gaussian filter.

The Gaussian coefficients are stored in CUDA constant memory.

Each CUDA thread computes one output pixel. Image boundaries are
handled by clamping neighboring coordinates to valid image positions.

## Dataset

The project uses a synthetic benchmark dataset containing 120 RGB
images.

Each image is 128x128 pixels.

The dataset contains gradients, circular patterns, checkerboard
patterns, sinusoidal patterns, and noisy images.

The dataset generator is included in the repository so the experiment
can be reproduced without downloading an external dataset.

## Project Structure

cuda_image_processing/

    src/main.cu
    generate_dataset.py
    validate.py
    Makefile
    run.sh
    input/
    output/
    results/

## Requirements

- NVIDIA GPU
- CUDA Toolkit
- nvcc
- Python 3

## Build

    make

## Generate Dataset

    python3 generate_dataset.py --count 120 --width 128 --height 128

## Run

    ./cuda_image_processor --input input --output output --sigma 1.5

Or:

    ./run.sh

## Validate

    python3 validate.py

## Command-Line Arguments

The CUDA application supports:

    --input <directory>
    --output <directory>
    --sigma <value>

Example:

    ./cuda_image_processor --input input --output output --sigma 1.5

## GPU Processing

The program allocates GPU memory once and reuses the device buffers
while processing the image batch.

For every image:

1. RGB data is copied from host memory to device memory.
2. The grayscale CUDA kernel processes the pixels.
3. The Gaussian blur CUDA kernel processes the grayscale image.
4. The processed result is copied back to host memory.
5. The output image is saved as a PGM file.

CUDA events are used to measure GPU pipeline execution time.

## Lessons Learned

This project demonstrates how image processing can be divided into
independent pixel-level operations that map naturally to CUDA threads.

One challenge was handling image boundaries during Gaussian filtering.
The solution was to clamp neighboring coordinates to valid image
coordinates.

Another important consideration was transferring image data between
host memory and device memory. Device buffers are allocated once and
reused for the complete image batch to avoid unnecessary repeated
allocation overhead.

The project provided practical experience with CUDA kernels, GPU
memory transfers, constant memory, CUDA events, grid and block
configuration, and batch image processing.

## Reproducibility

The complete experiment can be reproduced with:

    python3 generate_dataset.py --count 120 --width 128 --height 128
    make
    ./run.sh
    python3 validate.py
