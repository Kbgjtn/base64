#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-testdata}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEQUENTIAL_GENERATOR="$SCRIPT_DIR/generate-sequential.py"

# ---------------------------------------------------------------------------
# Corpus sizes
# ---------------------------------------------------------------------------
#
# Small sizes:
#
#   Base64 quantum boundaries:
#       1, 2, 3, 4
#       11, 12
#       15, 16, 17
#
#   Block/alignment boundaries:
#       31, 32, 33
#       63, 64, 65
#       127, 128, 129
#       255, 256, 257
#
# Large sizes:
#
#   Throughput:
#       1 KiB
#       4 KiB
#       16 KiB
#       64 KiB
#       256 KiB
#       1 MiB
#       4 MiB
#       16 MiB
#
#   Working-set / cache scaling:
#       32 MiB
#       64 MiB
#       128 MiB
#       256 MiB
#

SMALL_SIZES="
0:empty
1:1-byte
2:2-bytes
3:3-bytes
4:4-bytes

11:11-bytes
12:12-bytes

15:15-bytes
16:16-bytes
17:17-bytes

31:31-bytes
32:32-bytes
33:33-bytes

63:63-bytes
64:64-bytes
65:65-bytes

127:127-bytes
128:128-bytes
129:129-bytes

255:255-bytes
256:256-bytes
257:257-bytes
"

LARGE_SIZES="
1024:1-kib
4096:4-kib
16384:16-kib
65536:64-kib
262144:256-kib

1048576:1-mib
4194304:4-mib
16777216:16-mib

33554432:32-mib
67108864:64-mib
134217728:128-mib
268435456:256-mib
"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

die()
{
    echo "error: $*" >&2
    exit 1
}

require_command()
{
    command -v "$1" >/dev/null 2>&1 ||
        die "required command not found: $1"
}

# ---------------------------------------------------------------------------
# Zero-filled corpus
# ---------------------------------------------------------------------------

generate_zeros()
{
    local size=$1
    local output=$2

    if [ "$size" -eq 0 ]; then
        : > "$output"
        return
    fi

    local full_mib=$((size / 1048576))
    local remainder=$((size % 1048576))

    if [ "$full_mib" -gt 0 ]; then
        dd if=/dev/zero \
            of="$output" \
            bs=1M \
            count="$full_mib" \
            status=none
    else
        : > "$output"
    fi

    if [ "$remainder" -gt 0 ]; then
        dd if=/dev/zero \
            of="$output" \
            bs=1 \
            count="$remainder" \
            oflag=append \
            conv=notrunc \
            status=none
    fi
}

# ---------------------------------------------------------------------------
# Random corpus
# ---------------------------------------------------------------------------
#
# /dev/urandom is used only to create the fixtures.
#
# Once generated, commit the resulting files if you want benchmark runs
# across machines to use exactly the same random corpus.
#

generate_random()
{
    local size=$1
    local output=$2

    if [ "$size" -eq 0 ]; then
        : > "$output"
        return
    fi

    dd if=/dev/urandom \
        of="$output" \
        bs=1M \
        count="$size" \
        iflag=count_bytes \
        status=none
}

# ---------------------------------------------------------------------------
# Sequential corpus
# ---------------------------------------------------------------------------
#
# The Python helper generates:
#
#     00 01 02 ... ff 00 01 02 ... ff ...
#
# This is deterministic and works efficiently for very large fixtures.
#

generate_sequential()
{
    local size=$1
    local output=$2

    python3 "$SEQUENTIAL_GENERATOR" "$output" "$size"
}

# ---------------------------------------------------------------------------
# Generate one case
# ---------------------------------------------------------------------------

generate_case()
{
    local kind=$1
    local size=$2
    local name=$3

    local directory="$ROOT_DIR/$kind"
    local output="$directory/$name.bin"

    mkdir -p "$directory"

    case "$kind" in
        random)
            generate_random "$size" "$output"
            ;;
        zeros)
            generate_zeros "$size" "$output"
            ;;
        sequential)
            generate_sequential "$size" "$output"
            ;;
        *)
            die "unknown corpus type: $kind"
            ;;
    esac

    local actual_size
    actual_size=$(wc -c < "$output" | tr -d '[:space:]')

    if [ "$actual_size" -ne "$size" ]; then
        rm -f "$output"

        die \
            "size mismatch for $output: expected $size, got $actual_size"
    fi

    printf '  %-10s %-16s %12d bytes\n' \
        "$kind" \
        "$name" \
        "$size"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

require_command dd
require_command wc
require_command tr
require_command du
require_command python3

if [ ! -f "$SEQUENTIAL_GENERATOR" ]; then
    die "sequential generator not found: $SEQUENTIAL_GENERATOR"
fi

mkdir -p "$ROOT_DIR"

echo "Generating Base64 benchmark corpus..."
echo
echo "Output: $ROOT_DIR"
echo

# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

echo "== random =="

for spec in $SMALL_SIZES $LARGE_SIZES; do
    size=${spec%%:*}
    name=${spec#*:}

    generate_case random "$size" "$name"
done

echo

# ---------------------------------------------------------------------------
# Zeros
# ---------------------------------------------------------------------------

echo "== zeros =="

for spec in $LARGE_SIZES; do
    size=${spec%%:*}
    name=${spec#*:}

    generate_case zeros "$size" "$name"
done

echo

# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------

echo "== sequential =="

for spec in $LARGE_SIZES; do
    size=${spec%%:*}
    name=${spec#*:}

    generate_case sequential "$size" "$name"
done

echo
echo "Done."
echo

total_size=$(du -sh "$ROOT_DIR" | cut -f1)

echo "Total corpus size: $total_size"

