#!/usr/bin/env python3

import sys


CHUNK_SIZE = 1024 * 1024  # 1 MiB


def main() -> None:
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} OUTPUT SIZE",
            file=sys.stderr,
        )
        sys.exit(2)

    output = sys.argv[1]

    try:
        size = int(sys.argv[2])
    except ValueError:
        print(f"invalid size: {sys.argv[2]}", file=sys.stderr)
        sys.exit(2)

    if size < 0:
        print("size must be non-negative", file=sys.stderr)
        sys.exit(2)

    # One complete 256-byte sequence:
    #
    #   00 01 02 ... fd fe ff
    #
    pattern = bytes(range(256))

    # Build exactly 1 MiB of:
    #
    #   00 01 02 ... ff 00 01 02 ... ff ...
    #
    chunk = pattern * (CHUNK_SIZE // len(pattern))

    with open(output, "wb") as f:
        remaining = size

        while remaining >= len(chunk):
            f.write(chunk)
            remaining -= len(chunk)

        if remaining:
            # chunk contains enough data for any remainder < 1 MiB.
            f.write(chunk[:remaining])


if __name__ == "__main__":
    main()

