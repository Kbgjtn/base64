const std = @import("std");

pub const u8x16 = @Vector(16, u8);
pub const u8x32 = @Vector(32, u8);
pub const u8x64 = @Vector(64, u8);

pub const u6x32 = @Vector(32, u6);

pub const u16x8 = @Vector(8, u16);
pub const u16x16 = @Vector(16, u16);
pub const u16x32 = @Vector(32, u16);

pub const Tier = enum {
    avx2,
    simd128,
    scalar,

    pub fn fromCpu(cpu: std.Target.Cpu) Tier {
        return switch (cpu.arch) {
            .x86_64 => blk: {
                const features = cpu.features;
                const has_ssse3 = std.Target.x86.featureSetHas(features, .ssse3);
                const has_avx2 = std.Target.x86.featureSetHas(features, .avx2);

                // .avx2 requires ssse3 too: encodeChunk16 (SSSE3's pshufb) still runs whenever
                // tier != .scalar, including at the .avx2 tier, for the <24-byte remainder after
                // the main loop.
                //
                // Checking both bits here means that guarantee is enforced by this function,
                // not assumed from real-world x86 vendor history.
                break :blk if (has_avx2 and has_ssse3)
                    .avx2
                else if (has_ssse3)
                    .simd128
                else
                    .scalar;
            },

            // NEON is mandatory on AArch64.
            .aarch64 => .simd128,

            .wasm32, .wasm64 => if (std.Target.wasm.featureSetHas(cpu.features, .simd128))
                .simd128
            else
                .scalar,

            else => .scalar,
        };
    }

    test "avx2 tier always implies simd128-capable" {
        // Construct a target that reports avx2 but not ssse3.
        // If Tier ever regresses to returning .avx2 without confirming ssse3, this fails.
        var features = std.Target.Cpu.Feature.Set.empty;
        features.addFeature(@intFromEnum(std.Target.x86.Feature.avx2));
        // ssse3 deliberately left unset

        const cpu: std.Target.Cpu = .{
            .arch = .x86_64,
            .model = &std.Target.x86.cpu.generic,
            .features = features,
        };

        try std.testing.expect(Tier.fromCpu(cpu) != .avx2);
    }
};

const triple_pair_pattern = [_]u6{
    0, 0, 1, 0,
    2, 1, 2, 2,
};

const source_block_len = 12;
const lane_block_len = 16;

/// Returns indices that expand each byte triple `[a, b, c]` into the four
/// byte pairs `[a, a]`, `[b, a]`, `[c, b]`, and `[c, c]`.
///
/// The resulting pairs are ordered `[lo, hi]` for consumption by the
/// little-endian pair-bitcast operation. The caller is responsible for
/// ensuring little-endian target compatibility.
pub fn expandTriplePairs(comptime groups: usize) [groups * 8]i32 {
    var result: [groups * 8]i32 = undefined;

    for (0..groups) |group| {
        for (triple_pair_pattern, 0..) |offset, k| {
            const source = 3 * group + offset;
            const lane = source +
                (source / source_block_len) *
                    (lane_block_len - source_block_len);

            result[group * 8 + k] = @intCast(lane);
        }
    }

    return result;
}

const LoadSegment = struct {
    src: usize,
    dst: usize,
    len: usize,
};

/// Returns indices for combining two source segments into one output vector.
///
/// Segment `a` uses regular source indices. Segment `b` uses complemented
/// indices to identify its source lanes. Unused output lanes are set to zero.
pub fn loadIndices(
    comptime output_width: usize,
    comptime a: LoadSegment,
    comptime b: LoadSegment,
) [output_width]i32 {
    // Unused lanes are masked later; zero is a valid padding value.
    var indices: [output_width]i32 = .{0} ** output_width;

    for (0..a.len) |j| {
        indices[a.dst + j] = @intCast(a.src + j);
    }

    for (0..b.len) |j| {
        const source_index: i32 = @intCast(b.src + j);
        indices[b.dst + j] = ~source_index;
    }

    return indices;
}

/// Returns a vector of per-lane shift amounts by repeating `pattern`.
///
/// `lanes` must be divisible by 4.
pub fn laneShifts(
    comptime lanes: usize,
    comptime pattern: [4]u4,
) @Vector(lanes, u16) {
    comptime {
        if (lanes % 4 != 0) {
            @compileError("laneShifts: lane count must be divisible by 4");
        }

        const Vec = @Vector(lanes, u16);
        var result: Vec = undefined;

        for (0..lanes) |lane| {
            result[lane] = @as(u16, pattern[lane % 4]);
        }

        return result;
    }
}
