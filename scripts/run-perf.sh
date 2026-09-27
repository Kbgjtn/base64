#!/usr/bin/env bash
set -euo pipefail

CPU="${CPU:-3}"
BIN="${BIN:-./zig-out/bin/base64}"
RUNS="${RUNS:-10}"
OUT_DIR="${OUT_DIR:-perf-results}"

# zig build -Doptimize=ReleaseFast -Dlarge=true --summary all

mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "Benchmark executable not found or not executable: $BIN" >&2
    exit 1
fi

command -v taskset >/dev/null 2>&1 || {
    echo "taskset is required" >&2
    exit 1
}

command -v perf >/dev/null 2>&1 || {
    echo "perf is required" >&2
    exit 1
}

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
PREFIX="$OUT_DIR/bench-cpu${CPU}-${TIMESTAMP}"

echo "CPU:       $CPU"
echo "Binary:    $BIN"
echo "Stat runs: $RUNS"
echo "Output:    $PREFIX"
echo

echo "Running benchmark..."
taskset -c "$CPU" "$BIN" \
    | tee "${PREFIX}.txt"

echo
echo "Running perf stat..."

taskset -c "$CPU" \
perf stat \
    -r "$RUNS" \
    -e task-clock \
    -e context-switches \
    -e cpu-migrations \
    -e page-faults \
    -e cycles \
    -e instructions \
    -e branches \
    -e branch-misses \
    -e cache-references \
    -e cache-misses \
    -- "$BIN" \
    2> "${PREFIX}.stat.txt"

cat "${PREFIX}.stat.txt"

echo
echo "Running perf record..."

taskset -c "$CPU" \
perf record \
    -o "${PREFIX}.data" \
    -e cycles:u \
    --call-graph dwarf \
    -- "$BIN"

echo
echo "Generating perf report..."

perf report \
    --stdio \
    -i "${PREFIX}.data" \
    --sort comm,dso,symbol \
    > "${PREFIX}.report.txt"

cat "${PREFIX}.report.txt"

echo
echo "Files written:"
echo "  Benchmark output: ${PREFIX}.txt"
echo "  perf stat:        ${PREFIX}.stat.txt"
echo "  perf data:        ${PREFIX}.data"
echo "  perf report:      ${PREFIX}.report.txt"

