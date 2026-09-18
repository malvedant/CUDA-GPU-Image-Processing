import argparse
import math
import random
from pathlib import Path


def generate_image(index, width, height):
    rng = random.Random(10000 + index)
    mode = index % 5

    pixels = bytearray()

    cx = width / 2.0 + (index % 7 - 3) * 4.0
    cy = height / 2.0 + (index % 5 - 2) * 4.0

    for y in range(height):
        for x in range(width):
            nx = x / (width - 1)
            ny = y / (height - 1)

            if mode == 0:
                r = int(255 * nx)
                g = int(255 * ny)
                b = int(255 * ((nx + ny) / 2.0))

            elif mode == 1:
                distance = math.sqrt(
                    (x - cx) ** 2 + (y - cy) ** 2
                )

                value = int(
                    128
                    + 127
                    * math.sin(
                        distance / 5.0 + index * 0.15
                    )
                )

                r = value
                g = int(255 - value)
                b = int((value + index * 3) % 256)

            elif mode == 2:
                checker = (
                    ((x // 8) + (y // 8) + index) % 2
                )

                if checker:
                    r, g, b = 230, 230, 230
                else:
                    r, g, b = 30, 60, 120

            elif mode == 3:
                r = int(
                    127
                    + 127
                    * math.sin(
                        x / 8.0 + index * 0.2
                    )
                )

                g = int(
                    127
                    + 127
                    * math.sin(
                        y / 9.0 + index * 0.15
                    )
                )

                b = int(
                    127
                    + 127
                    * math.sin(
                        (x + y) / 13.0
                    )
                )

            else:
                base = int(
                    255
                    * (
                        0.5 * nx
                        + 0.5 * ny
                    )
                )

                noise = rng.randint(-35, 35)

                r = max(0, min(255, base + noise))
                g = max(0, min(255, base + noise + 20))
                b = max(0, min(255, base + noise - 20))

            pixels.extend(
                (
                    max(0, min(255, r)),
                    max(0, min(255, g)),
                    max(0, min(255, b)),
                )
            )

    return pixels


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic RGB PPM images."
    )

    parser.add_argument(
        "--count",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--output",
        default="input",
    )

    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for index in range(args.count):
        filename = output / f"image_{index + 1:03d}.ppm"

        pixels = generate_image(
            index,
            args.width,
            args.height,
        )

        with open(filename, "wb") as file:
            file.write(
                f"P6\n{args.width} {args.height}\n255\n"
                .encode("ascii")
            )
            file.write(pixels)

    print(
        f"Generated {args.count} images "
        f"({args.width}x{args.height}) in {output}"
    )


if __name__ == "__main__":
    main()
