from pathlib import Path


INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

inputs = sorted(INPUT_DIR.glob("*.ppm"))
outputs = sorted(OUTPUT_DIR.glob("*_processed.pgm"))

print("CUDA IMAGE PROCESSING VALIDATION")
print("--------------------------------")
print(f"Input images : {len(inputs)}")
print(f"Output images: {len(outputs)}")

if len(inputs) != 120:
    raise SystemExit(
        f"Expected 120 input images, found {len(inputs)}"
    )

if len(outputs) != len(inputs):
    raise SystemExit(
        "Output count does not match input count."
    )

for output in outputs:
    with open(output, "rb") as file:
        magic = file.readline().strip()
        dimensions = file.readline().split()
        maximum = file.readline().strip()
        data = file.read()

    if magic != b"P5":
        raise SystemExit(
            f"Invalid PGM file: {output}"
        )

    width = int(dimensions[0])
    height = int(dimensions[1])

    if maximum != b"255":
        raise SystemExit(
            f"Invalid maximum value: {output}"
        )

    if len(data) != width * height:
        raise SystemExit(
            f"Invalid pixel count: {output}"
        )

print("--------------------------------")
print("Validation: SUCCESS")
print("All 120 input images have matching outputs.")
