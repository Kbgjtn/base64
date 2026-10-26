#!/usr/bin/env bash
#
# Build and run the benchmark once per x86_64 SIMD tier, pinned to one core.
#
#   scripts/bench-tiers.sh                  # all tiers, default settings
#   TIERS="x86_64_v3" RUNS=9 scripts/bench-tiers.sh
#   PERF=1 scripts/bench-tiers.sh           # also perf stat/record per tier
#   LABEL="cloud VM" scripts/bench-tiers.sh # free-text note stored in meta.txt
#
# Output: bench/results/<tier>-<cpu-slug>/{bench.csv,bench.txt,meta.txt[,perf/]}
# Then:   python3 scripts/bench-summary.py bench/results > bench/results/SUMMARY.md
#
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TIERS="${TIERS:-x86_64 x86_64_v2 x86_64_v3}"
CPU="${CPU:-2}"                       # core to pin to (taskset -c)
RUNS="${RUNS:-7}"
TARGET_MS="${TARGET_MS:-10}"
CORPUS="${CORPUS:-bench/corpus}"
OUT_ROOT="${OUT_ROOT:-bench/results}"
PERF="${PERF:-0}"
LABEL="${LABEL:-}"
ZIG="${ZIG:-zig}"
EXTRA_ARGS=("$@")                     # forwarded to the bench binary

die() { echo "error: $*" >&2; exit 1; }

command -v "$ZIG" >/dev/null || die "zig not found (set ZIG=...)"
command -v taskset >/dev/null || die "taskset is required (util-linux)"
[[ -f "$CORPUS/manifest.tsv" ]] || die "no corpus at $CORPUS; run: python3 scripts/gen-corpus.py -o $CORPUS"
if [[ "$PERF" == 1 ]]; then command -v perf >/dev/null || die "PERF=1 but perf is not installed"; fi

python3 scripts/gen-corpus.py -o "$CORPUS" --verify >/dev/null || die "corpus checksum mismatch; regenerate it"

cpu_model="$(awk -F': *' '/^model name/ { print $2; exit }' /proc/cpuinfo 2>/dev/null || true)"
cpu_model="${cpu_model:-$(uname -m)}"
cpu_slug="$(printf '%s' "$cpu_model" | tr '[:upper:]' '[:lower:]' | sed -E 's/\((r|tm)\)//g; s/(cpu|processor|@.*)//g; s/[^a-z0-9]+/-/g; s/^-+|-+$//g')"

governor="$(cat "/sys/devices/system/cpu/cpu${CPU}/cpufreq/scaling_governor" 2>/dev/null || echo n/a)"
if [[ -r /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    turbo="no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
elif [[ -r /sys/devices/system/cpu/cpufreq/boost ]]; then
    turbo="boost=$(cat /sys/devices/system/cpu/cpufreq/boost)"
else
    turbo=n/a
fi
virt="$(systemd-detect-virt 2>/dev/null)" || true
if [[ -z "$virt" || "$virt" == none ]]; then
    grep -q '^flags.* hypervisor' /proc/cpuinfo && virt=hypervisor || virt=none
fi
git_rev="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo -dirty)"

for tier in $TIERS; do
    out="$OUT_ROOT/$tier-$cpu_slug"
    prefix="zig-out/bench-$tier"
    mkdir -p "$out"

    echo "== $tier -> $out"
    "$ZIG" build bench -Doptimize=ReleaseFast -Dcpu="$tier" -p "$prefix" 2>&1 | grep -v '^$' || true
    bin="$prefix/bin/bench"
    [[ -x "$bin" ]] || die "build failed for $tier"

    {
        echo "date_utc:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "label:      ${LABEL:-}"
        echo "tier_cpu:   $tier (-Dcpu, ReleaseFast)"
        echo "host_cpu:   $cpu_model"
        echo "cores:      $(nproc 2>/dev/null || echo ?) (pinned to $CPU)"
        echo "virt:       $virt"
        echo "governor:   $governor"
        echo "turbo:      $turbo"
        echo "kernel:     $(uname -srm)"
        echo "zig:        $("$ZIG" version)"
        echo "git_rev:    $git_rev"
        echo "corpus:     $CORPUS ($(sha256sum "$CORPUS/SHA256SUMS" | cut -c1-16))"
        echo "args:       --runs $RUNS --target-ms $TARGET_MS ${EXTRA_ARGS[*]:-}"
    } > "$out/meta.txt"

    taskset -c "$CPU" "$bin" --corpus "$CORPUS" --runs "$RUNS" --target-ms "$TARGET_MS" \
        --csv "${EXTRA_ARGS[@]}" > "$out/bench.csv"
    echo "   $(($(wc -l < "$out/bench.csv") - 1)) rows -> $out/bench.csv"

    if [[ "$PERF" == 1 ]]; then
        CPU="$CPU" BIN="$bin" OUT_DIR="$out/perf" RUNS=3 \
            scripts/run-perf.sh --corpus "$CORPUS" --runs 3 "${EXTRA_ARGS[@]}" > "$out/perf.log" 2>&1 \
            || echo "   perf failed, see $out/perf.log"
    fi
done

echo
echo "Summary: python3 scripts/bench-summary.py $OUT_ROOT"
