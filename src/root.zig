//! Base64 encoding and decoding, as specified by
//! [RFC 4648](https://datatracker.ietf.org/doc/html/rfc4648).
//!
//! Base64 works in 3-byte (24-bit) blocks, split into 4 groups of 6 bits.
//! Each 6-bit group maps to one character of a 64-character alphabet:
//!   0-25  => uppercase ASCII letters [A-Z]
//!   26-51 => lowercase ASCII letters [a-z]
//!   52-61 => digits [0-9]
//!   62-63 => `+` and `/`
//!
//! When the input length isn't a multiple of 3 bytes, the output is padded
//! with `=` so that the encoded length is always a multiple of 4:
//!   * 1 leftover byte  = 2 chars + "=="
//!   * 2 leftover bytes = 3 chars + "="
//!
//! 3 bytes were chosen as the block size because it's the smallest number
//! of bytes whose total bit count (n * 8) is evenly divisible by 6:
//! the smallest n solving `8n % 6 == 0` is n = 3, giving 24 bits => 4
//! six-bit characters.

const std = @import("std");
const builtin = @import("builtin");

const native_endian = builtin.target.cpu.arch.endian();

// IMPORTANT! Pair bit-casts require little-endian byte order.
//
// `pairs()` stores each 16-bit value as [low_byte, high_byte].
// The @bitCast operation below relies on little-endian layout:
//
// [lo, hi] => (hi << 8) | lo
//
// On a big-endian target, the bytes would be interpreted in the opposite
// order, it'd producing incorrect 6-bit values without causing a runtime error.
//
// A compile-time check rejects big-endian targets. If big-endian support is
// added later, generate `pair_idx` and `pair_idx16` with [hi, lo] ordering
// when `builtin.target.cpu.arch.endian() == .big`, then test the results under
// a big-endian emulator such as QEMU.
comptime {
    if (!(native_endian == .little)) {
        @compileError("pair bit-cast operations require a little-endian target");
    }
}

const simd = @import("simd.zig");

/// Computed once, at comptime, from the actual compilation target, not re-derived here.
/// See simd_tier.zig; build.zig calls the same function against the resolved build target
/// so the two can't drift apart.
const tier: simd.Tier = .fromCpu(builtin.cpu);

/// One 32-byte input/output chunk.
const Chunk = simd.u8x32;

/// One 16-byte input/output chunk.
const Chunk16 = simd.u8x16;

/// Sixteen little-endian u16 words, each containing one [lo, hi] pair.
const Pairs16 = simd.u16x16;

/// Thirty-two little-endian u16 words, each containing one [lo, hi] pair.
const Pairs = simd.u16x32;

const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const pair_indices_8_groups = simd.expandTriplePairs(8);
const pair_indices_4_groups = simd.expandTriplePairs(4);

/// Maps two Base64 sextets to two packed ASCII bytes.
///
/// Each entry is stored as a little-endian u16 so the bytes are written in
/// alphabet order by `writeInt(u16, ..., .little)`.
const pair_lut: [4096]u16 = blk: {
    @setEvalBranchQuota(1 << 20);
    var t: [4096]u16 = undefined;
    for (&t, 0..) |*e, i| {
        e.* = @as(u16, alphabet[i >> 6]) |
            (@as(u16, alphabet[i & 0x3f]) << 8);
    }
    break :blk t;
};

/// Writes the two Base64 characters encoded by `sextets` to `dest[offset..]`.
///
/// The upper 6 bits select the first character and the lower 6 bits select
/// the second character.
fn writeScalarPair(dest: []u8, i: usize, p: u12) void {
    std.mem.writeInt(u16, dest[i..][0..2], pair_lut[p], .little);
}

// Combines two overlapping 16-byte loads into a 32-byte chunk.
//
// A contributes source[i .. i+12] at chunk[0 .. 12].
// B contributes source[i+12 .. i+24] at chunk[16 .. 28].
// The remaining lanes are padding.
const load_indices_lanes = simd.loadIndices(
    32,
    .{ .src = 0, .dst = 0, .len = 12 },
    .{ .src = 4, .dst = 16, .len = 12 },
);

// Combines two overlapping 8-byte loads into a 16-byte chunk.
//
// A contributes source[i .. i+8] at chunk[0 .. 8].
// B contributes source[i+8 .. i+12] at chunk[8 .. 12].
// The remaining lanes are padding.
const load_indices_16_lanes = simd.loadIndices(
    16,
    .{ .src = 0, .dst = 0, .len = 8 },
    .{ .src = 4, .dst = 8, .len = 4 },
);

const shifts_32_lanes = simd.laneShifts(32, .{ 10, 4, 6, 0 });
const shifts_16_lanes = simd.laneShifts(16, .{ 10, 4, 6, 0 });

/// Converts one Base64 sextet in each lane to its ASCII character.
///
/// Every input lane must contain a value in the range `0..63`.
fn sextetsToAscii(comptime lanes: usize, v: @Vector(lanes, u8)) @Vector(lanes, u8) {
    const Vec = @Vector(lanes, u8);
    const zero: Vec = @splat(0);

    var offset: Vec = @splat(65); // 'A'

    // 26..51: move from 'A'..'Z' to 'a'..'z'.
    offset +%= @select(
        u8,
        v > @as(Vec, @splat(25)),
        @as(Vec, @splat(6)),
        zero,
    );

    // 52..63: move to the numeric/symbol range.
    offset -%= @select(
        u8,
        v > @as(Vec, @splat(51)),
        @as(Vec, @splat(75)),
        zero,
    );

    // 62 -> '+', 63 -> '/'.
    // The low bit distinguishes 62 from 63.
    const tail: Vec =
        (v & @as(Vec, @splat(1))) *% @as(Vec, @splat(3));

    offset -%= @select(
        u8,
        v > @as(Vec, @splat(61)),
        @as(Vec, @splat(15)) -% tail,
        zero,
    );

    return v +% offset;
}

/// Encodes 24 input bytes starting at `i` into 32 Base64 bytes.
///
/// Uses little-endian pair bit-casts. Big-endian targets are rejected at
/// compile time.
inline fn encodeChunk(source: []const u8, i: usize) Chunk {
    const a: Chunk16 = source[i..][0..16].*;
    const b: Chunk16 = source[i + 8 ..][0..16].*;

    const chunk: Chunk = @shuffle(u8, a, b, load_indices_lanes);

    // Each pair is laid out as [lo, hi].
    // On little-endian targets, @bitCast interprets this as:
    //
    //     w = (hi << 8) | lo
    const pairs: @Vector(64, u8) = @shuffle(
        u8,
        chunk,
        undefined,
        pair_indices_8_groups,
    );
    const w: Pairs = @bitCast(pairs);
    const v: Chunk = @truncate(
        (w >> shifts_32_lanes) & @as(Pairs, @splat(0x3f)),
    );

    return sextetsToAscii(32, v);
}

/// Encodes 12 input bytes starting at `i` into 16 Base64 bytes.
///
/// Uses little-endian pair bit-casts. Big-endian targets are rejected at
/// compile time.
inline fn encodeChunk16(source: []const u8, i: usize) Chunk16 {
    const a: @Vector(8, u8) = source[i..][0..8].*;
    const b: @Vector(8, u8) = source[i + 4 ..][0..8].*;

    const chunk: Chunk16 = @shuffle(u8, a, b, load_indices_16_lanes);

    // `pair_idx16` produces [low_byte, high_byte] pairs.
    // The second shuffle operand is unused, but must still be defined.
    const zero: Chunk16 = @splat(0);
    const pairs: @Vector(32, u8) = @shuffle(
        u8,
        chunk,
        zero,
        pair_indices_4_groups,
    );
    const w: Pairs16 = @bitCast(pairs);
    const v: Chunk16 = @truncate(
        (w >> shifts_16_lanes) & @as(Pairs16, @splat(0x3f)),
    );

    return sextetsToAscii(16, v);
}

/// Returns the padded Base64 output size.
///
/// `source_len` must be small enough for the padded output to fit in `usize`.
pub fn computeSize(source_len: usize) usize {
    const groups = source_len / 3 + @intFromBool(source_len % 3 != 0);
    return groups * 4;
}

test "computeSize: does not overflow for representable outputs" {
    const max_len = (std.math.maxInt(usize) / 4) * 3;
    _ = computeSize(max_len);
}

/// Returns the exact decoded size for a padded Base64 slice.
///
/// Returns `error.InvalidPadding` if:
/// - the length is not a multiple of four;
/// - padding is not at the end; or
/// - more than two padding characters are present.
///
/// NOTE The base64 alphabet is not validated.
pub fn computeSizeForSlice(source: []const u8) !usize {
    const len = source.len;

    if (len % 4 != 0) {
        return error.InvalidPadding;
    }

    if (len == 0) {
        return 0;
    }

    var padding: usize = 0;
    if (source[len - 1] == '=') {
        padding += 1;
    }

    if (len >= 2 and source[len - 2] == '=') {
        padding += 1;
    }

    // Any '=' before the final padding positions is invalid.
    const non_padding_len = len - padding;
    if (std.mem.findScalar(u8, source[0..non_padding_len], '=') != null) {
        return error.InvalidPadding;
    }

    return (len / 4) * 3 - padding;
}

// TODO!
// 1. Blocking for std upstreaming: alphabet/pad parameterization and the Codecs shape std's
// contract is per-variant instances.
// 2. Nice to have: encodeWriter.

/// A Base64 encoder that supports both buffer-based and allocating encoding.
pub const Encoder = struct {
    /// Encodes `source` into `dest` and returns the number of bytes written.
    ///
    /// The `dest` must be at least `computeSize(source.len)` bytes long.
    /// The required size can be computed with `computeSize` before calling.
    ///
    /// Passing a smaller destinatin is a programming error and triggers an
    /// assertion failure in safety-enabled builds. The `dest` maby be larger
    /// than required and can be reused as an oversized scratch buffer.
    pub fn encodeToSlice(dest: []u8, source: []const u8) usize {
        const dest_size = computeSize(source.len);
        std.debug.assert(dest.len >= dest_size);

        const source_len = source.len;
        if (source_len == 0) {
            return 0;
        }

        var source_index: usize = 0;
        var dest_index: usize = 0;

        // ymm main loop: 24 bytes -> 32 chars.
        if (tier == .avx2 and source_len >= 24) {
            while (source_index + 24 <= source_len) : ({
                source_index += 24;
                dest_index += 32;
            }) {
                dest[dest_index..][0..32].* = encodeChunk(source, source_index);
            }
        }

        // xmm half-chunk: 12 bytes => 16 chars for the 12..23 byte band.
        // Runs at most once (remaining < 24 here); the @Vector(16) ops
        // stay in xmm registers with no lane-crossing at all.
        if (tier != .scalar and source_index + 12 <= source_len) {
            while (source_index + 12 <= source_len) : ({
                source_index += 12;
                dest_index += 16;
            }) {
                dest[dest_index..][0..16].* = encodeChunk16(source, source_index);
            }
        } else {
            // Scalar bulk path: 12 bytes -> 16 chars via one u128 load
            // split into independent u64/u32 shift chains feeding the
            // 12-bit pair LUT. Reads 16 bytes, uses 12.
            while (source_index + 15 < source_len) : ({
                source_index += 12;
                dest_index += 16;
            }) {
                const bits = std.mem.readInt(
                    u128,
                    source[source_index..][0..16],
                    .big,
                );

                // Big-endian read: source bytes 0..11 occupy bits 127..32;
                // bytes 12..15 in the low 32 bits are discarded.
                const hi: u64 = @truncate(bits >> 64); // source bytes 0..7
                const mid: u32 = @truncate(bits >> 32); // source bytes 8..11

                // hi chain: 5 pairs + 4 leftover bits
                writeScalarPair(dest, dest_index + 0, @truncate(hi >> 52));
                writeScalarPair(dest, dest_index + 2, @truncate(hi >> 40));
                writeScalarPair(dest, dest_index + 4, @truncate(hi >> 28));
                writeScalarPair(dest, dest_index + 6, @truncate(hi >> 16));
                writeScalarPair(dest, dest_index + 8, @truncate(hi >> 4));

                // boundary pair: hi bits 3..0 ++ mid bits 31..24
                writeScalarPair(
                    dest,
                    dest_index + 10,
                    @truncate((@as(u64, hi & 0xf) << 8) | (mid >> 24)),
                );

                // mid chain: bytes 9..11
                writeScalarPair(dest, dest_index + 12, @truncate(mid >> 12));
                writeScalarPair(dest, dest_index + 14, @truncate(mid));
            }
        }

        // Remaining full triples (0..3 at every tier; arbitrary run only
        // if the u128 loop couldn't start, i.e. len < 15 handled below too:
        // this loop also covers whatever the block loops left behind).
        while (source_index + 3 <= source_len) : ({
            source_index += 3;
            dest_index += 4;
        }) {
            const bits = std.mem.readInt(
                u24,
                source[source_index..][0..3],
                .big,
            );
            dest[dest_index + 0] =
                alphabet[@truncate((bits >> 18) & 0x3f)];

            dest[dest_index + 1] =
                alphabet[@truncate((bits >> 12) & 0x3f)];

            dest[dest_index + 2] =
                alphabet[@truncate((bits >> 6) & 0x3f)];

            dest[dest_index + 3] =
                alphabet[@truncate(bits & 0x3f)];
        }

        // Encode the remaining zero, one, or two bytes.
        const remaining = source_len - source_index;
        switch (remaining) {
            0 => {},

            1 => {
                const first = source[source_index];
                dest[dest_index + 0] = alphabet[first >> 2];
                dest[dest_index + 1] = alphabet[(first & 0x03) << 4];
                dest[dest_index + 2] = '=';
                dest[dest_index + 3] = '=';
            },

            2 => {
                const first = source[source_index];
                const second = source[source_index + 1];

                dest[dest_index + 0] = alphabet[first >> 2];
                dest[dest_index + 1] = alphabet[((first & 0x03) << 4) | (second >> 4)];
                dest[dest_index + 2] = alphabet[(second & 0x0f) << 2];
                dest[dest_index + 3] = '=';
            },

            else => unreachable,
        }

        return dest_size;
    }

    /// Encodes `source` into a newly allocated buffer.
    ///
    /// The caller owns the returned slice and must free it with `gpa`.
    ///
    /// Returns an allocator error if the output buffer cannot be allocated.
    pub fn encode(gpa: std.mem.Allocator, source: []const u8) ![]u8 {
        const out = try gpa.alloc(u8, computeSize(source.len));
        _ = encodeToSlice(out, source);
        // return out directly since n == computeSize(source.len) == out.len always.
        return out;
    }
};

/// A Base64 decoder that supports both buffer-based and allocating encoding.
pub const Decoder = struct {
    const invalid_char_tst: u32 = 0xFF000000;
    const invalid_char: u8 = 0xFF;

    const base64_alphabet_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    /// Four pre-shifted 256-entry lookup tables, one per byte-position
    /// within a 4-character block. Layout borrowed from
    /// `std.base64.Base64Decoder.init`.
    const tables = blk: {
        var t: [4][256]u32 = .{[_]u32{invalid_char_tst} ** 256} ** 4;
        for (base64_alphabet_chars, 0..) |c, i| {
            const ci: u32 = @intCast(i);
            t[0][c] = ci << 2;
            t[1][c] = (ci >> 4) | ((ci & 0x0f) << 12);
            t[2][c] = ((ci & 0x3) << 22) | ((ci & 0x3c) << 6);
            t[3][c] = ci << 16;
        }
        break :blk t;
    };

    /// Compact 256-byte table used only by the last-block (padding-aware)
    /// path, so single-block/small decodes don't have to touch the full
    /// 4 KiB of `tables` — keeps the hot cache set small for that path.
    const char_to_index: [256]u8 = blk: {
        var t = [_]u8{invalid_char} ** 256;
        for (base64_alphabet_chars, 0..) |c, i| {
            t[c] = @intCast(i);
        }
        break :blk t;
    };

    pub const Error = error{
        InvalidCharacter,
        InvalidPadding,
        NoSpaceLeft,
        OutOfMemory,
    };

    pub const NonCanonicalPolicy = enum { reject, allow };

    /// Decodes `source` into `dest`. `dest` must be at least
    /// `computeSizeForSlice(source)` bytes long.
    pub fn decodeToSlice(dest: []u8, source: []const u8, policy: NonCanonicalPolicy) Error!usize {
        if (source.len % 4 != 0) {
            @branchHint(.unlikely);
            return error.InvalidPadding;
        }

        const dest_size = try computeSizeForSlice(source);
        if (dest_size == 0) return 0;
        if (dest.len < dest_size) {
            @branchHint(.unlikely);
            return error.NoSpaceLeft;
        }

        return decodeKnownSize(dest, source, dest_size, policy);
    }

    /// Same as `decodeToSlice` but skips recomputing `dest_size` when the
    /// caller already knows it (used by `decode` to avoid a duplicate
    /// `computeSizeForSlice` pass over `source`).
    fn decodeKnownSize(dest: []u8, source: []const u8, dest_size: usize, policy: NonCanonicalPolicy) Error!usize {
        const full_blocks = source.len / 4;
        if (full_blocks == 0) return 0;
        // Everything except the last block is padding-free and can be
        // decoded with the batched tables below; the last block is always
        // handled separately since it may carry padding.
        const bulk_len = (full_blocks - 1) * 4;

        var src_idx: usize = 0;
        var dst_idx: usize = 0;

        // Tier 1: 16 source bytes -> 12 dest bytes per iteration.
        while (src_idx + 16 <= bulk_len and dst_idx + 15 < dest.len) : ({
            src_idx += 16;
            dst_idx += 12;
        }) {
            var bits: u128 = 0;
            inline for (0..4) |i| {
                const b = src_idx + i * 4;
                var word: u128 = tables[0][source[b]];
                word |= tables[1][source[b + 1]];
                word |= tables[2][source[b + 2]];
                word |= tables[3][source[b + 3]];
                if ((word & invalid_char_tst) != 0) {
                    @branchHint(.unlikely);
                    return error.InvalidCharacter;
                }
                bits |= word << (24 * i);
            }
            std.mem.writeInt(u128, dest[dst_idx..][0..16], bits, .little);
        }

        // Tier 2: 4 source bytes -> 3 dest bytes per iteration.
        while (src_idx + 4 <= bulk_len and dst_idx + 3 < dest.len) : ({
            src_idx += 4;
            dst_idx += 3;
        }) {
            var bits = tables[0][source[src_idx]];
            bits |= tables[1][source[src_idx + 1]];
            bits |= tables[2][source[src_idx + 2]];
            bits |= tables[3][source[src_idx + 3]];
            if ((bits & invalid_char_tst) != 0) {
                @branchHint(.unlikely);
                return error.InvalidCharacter;
            }
            std.mem.writeInt(u32, dest[dst_idx..][0..4], bits, .little);
        }

        // `dest` is always allocated to the exact `dest_size` by both
        // callers (`decode` and `decodeToSlice`), so by this point the
        // tier-1 or tier-2 loops above have always fully consumed `bulk_len`
        // before `dst_idx` could become the limiting factor. This assert
        // documents that invariant instead of silently relying on it; if
        // it ever fires, a caller is passing an oversized `dest` with
        // trailing slack that this function was not designed to handle.
        std.debug.assert(src_idx == bulk_len);

        // Scalar fallback for any leftover full blocks (only reachable if
        // `dest`'s tail is too short for the batched writes above).
        // while (src_idx < bulk_len) : ({
        //     src_idx += 4;
        //     dst_idx += 3;
        // }) {
        //     var bits = tables[0][source[src_idx]];
        //     bits |= tables[1][source[src_idx + 1]];
        //     bits |= tables[2][source[src_idx + 2]];
        //     bits |= tables[3][source[src_idx + 3]];
        //     if ((bits & invalid_char_tst) != 0) return error.InvalidCharacter;
        //     dest[dst_idx + 0] = @truncate(bits & 0xFF);
        //     dest[dst_idx + 1] = @truncate((bits >> 8) & 0xFF);
        //     dest[dst_idx + 2] = @truncate((bits >> 16) & 0xFF);
        // }

        // Last block: may contain padding, so it's handled scalar and
        // separately from the bulk loops above.
        {
            const i = bulk_len;
            const c1 = source[i + 0];
            const c2 = source[i + 1];
            const c3 = source[i + 2];
            const c4 = source[i + 3];

            if (c1 == '=' or c2 == '=') {
                @branchHint(.unlikely);
                return error.InvalidPadding;
            }

            const v1 = char_to_index[c1];
            const v2 = char_to_index[c2];
            if (v1 == invalid_char or v2 == invalid_char) {
                @branchHint(.unlikely);
                return error.InvalidCharacter;
            }

            const is_c3_padded = c3 == '=';
            const is_c4_padded = c4 == '=';
            if (is_c3_padded and !is_c4_padded) {
                @branchHint(.unlikely);
                return error.InvalidPadding;
            }

            const bytes_to_write: usize = if (is_c3_padded) 1 else if (is_c4_padded) 2 else 3;

            var v3: u8 = 0;
            var v4: u8 = 0;
            if (!is_c3_padded) {
                v3 = char_to_index[c3];
                if (v3 == invalid_char) {
                    @branchHint(.unlikely);
                    return error.InvalidCharacter;
                }
            }
            if (!is_c4_padded) {
                v4 = char_to_index[c4];
                if (v4 == invalid_char) {
                    @branchHint(.unlikely);
                    return error.InvalidCharacter;
                }
            }

            const block: u32 =
                (@as(u32, v1) << 18) |
                (@as(u32, v2) << 12) |
                (@as(u32, v3) << 6) |
                (@as(u32, v4));

            if (policy == .reject) {
                if (bytes_to_write == 1 and (block & 0xFFFF) != 0) {
                    @branchHint(.unlikely);
                    return error.InvalidPadding;
                }
                if (bytes_to_write == 2 and (block & 0xFF) != 0) {
                    @branchHint(.unlikely);
                    return error.InvalidPadding;
                }
            }

            if (bytes_to_write >= 1) dest[dst_idx + 0] = @truncate((block >> 16) & 0xFF);
            if (bytes_to_write >= 2) dest[dst_idx + 1] = @truncate((block >> 8) & 0xFF);
            if (bytes_to_write == 3) dest[dst_idx + 2] = @truncate(block & 0xFF);

            dst_idx += bytes_to_write;
        }

        return dest_size;
    }

    pub fn decode(gpa: std.mem.Allocator, source: []const u8, policy: NonCanonicalPolicy) Error![]u8 {
        const size = try computeSizeForSlice(source);
        const out = try gpa.alloc(u8, size);
        errdefer gpa.free(out);
        const n = try decodeKnownSize(out, source, size, policy);
        return out[0..n];
    }
};

fn roundtrip(allocator: std.mem.Allocator, input: []const u8, policy: Decoder.NonCanonicalPolicy) !void {
    const encoded_len = computeSize(input.len);
    const encoded = try allocator.alloc(u8, encoded_len);
    defer allocator.free(encoded);

    const encoded_n = Encoder.encodeToSlice(encoded, input);
    try std.testing.expectEqual(encoded_len, encoded_n);

    const decoded_len = try computeSizeForSlice(encoded);
    const decoded = try allocator.alloc(u8, decoded_len);
    defer allocator.free(decoded);

    const written = try Decoder.decodeToSlice(decoded, encoded, policy);
    try std.testing.expectEqual(input.len, written);
    try std.testing.expectEqualSlices(u8, input, decoded);

    const decoded2 = try Decoder.decode(allocator, encoded, policy);
    defer allocator.free(decoded2);
    try std.testing.expectEqual(input.len, decoded2.len);
    try std.testing.expectEqualSlices(u8, input, decoded2);
}

fn expectMatchesStd(
    allocator: std.mem.Allocator,
    source: []const u8,
) !void {
    const expected_size =
        std.base64.standard.Encoder.calcSize(source.len);

    const expected_buffer =
        try allocator.alloc(u8, expected_size);
    defer allocator.free(expected_buffer);

    const got_buffer =
        try allocator.alloc(u8, expected_size);
    defer allocator.free(got_buffer);

    const expected =
        std.base64.standard.Encoder.encode(
            expected_buffer,
            source,
        );

    const got_len = Encoder.encodeToSlice(
        got_buffer,
        source,
    );

    const got = got_buffer[0..got_len];

    try std.testing.expectEqual(expected.len, got.len);
    try std.testing.expectEqualSlices(u8, expected, got);
}

fn checkRandomInput(
    allocator: std.mem.Allocator,
    random: std.Random,
    len: usize,
) !void {
    const source = try allocator.alloc(u8, len);
    defer allocator.free(source);

    random.bytes(source);

    try expectMatchesStd(allocator, source);
}

test "vector encoder matches std for dense short inputs" {
    const gpa = std.testing.allocator;

    // Boundary lengths: below/at/above the 24-byte vector step,
    // the 96-byte unrolled step (if present), and the scalar tail.
    const lengths = [_]usize{
        0,  1,  2,  3,  4,  5,
        11, 12, 15, 16, 17,
        22, 23, 24, 25, 26, // vector loop boundary
        46, 47, 48, 49, 50, // 2× chunk boundary
        94,  95,  96,  97, 98, // 4× chunk boundary (if unrolled)
        119, 120, 121,
    };

    var prng = std.Random.DefaultPrng.init(0xdeadbeef);
    const rand = prng.random();

    for (lengths) |len| {
        try checkRandomInput(gpa, rand, len);
    }
}

test "vector encoder matches std for lengths 0 through 100" {
    const gpa = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(0xdeadbeef);
    const random = prng.random();

    for (0..101) |len| {
        try checkRandomInput(gpa, random, len);
    }
}

test "vector encoder matches std on random large buffers" {
    const gpa = std.testing.allocator;

    const sizes = [_]usize{
        1024,
        4095,
        4096,
        4097,
        65536,
        1 << 20,
    };

    var prng = std.Random.DefaultPrng.init(0x12345678);
    const random = prng.random();

    for (sizes) |len| {
        for (0..4) |_| {
            try checkRandomInput(gpa, random, len);
        }
    }
}
test "vector encoder matches std on boundary lengths" {
    const gpa = std.testing.allocator;

    const sizes = [_]usize{
        1024,
        4095,
        4096,
        4097,
        65536,
        1 << 20,
    };

    var prng = std.Random.DefaultPrng.init(0x12345678);
    const rand = prng.random();

    for (sizes) |len| {
        try checkRandomInput(gpa, rand, len);
    }
}

test "Base64 encode/decode roundtrip and boundary sizes" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xdeadbeef);
    const random = prng.random();

    // Exhaustively cover every size around the encoder/decoder tier
    // boundaries (3/4-byte scalar tail, 12/16-byte batched tiers), rather
    // than relying on random sampling to eventually hit them.
    var size: usize = 0;
    while (size <= 40) : (size += 1) {
        const input = try allocator.alloc(u8, size);
        defer allocator.free(input);
        random.bytes(input);

        try roundtrip(allocator, input, .reject);
    }
}

test "Encoder known vectors" {
    // One test per possible byte-remainder (0, 1, 2 bytes leftover after
    // 3-byte grouping), matching std.base64's own hand-picked vectors.
    const cases = .{
        .{ "", "" },
        .{ "f", "Zg==" },
        .{ "fo", "Zm8=" },
        .{ "foo", "Zm9v" },
        .{ "foob", "Zm9vYg==" },
        .{ "fooba", "Zm9vYmE=" },
        .{ "foobar", "Zm9vYmFy" },
        // Long enough to cross the tier-1 (16 source bytes) batch boundary.
        .{ "foobarfoobarfoobarfoobarfoobarfoo", "Zm9vYmFyZm9vYmFyZm9vYmFyZm9vYmFyZm9vYmFyZm9v" },
    };

    inline for (cases) |case| {
        const input: []const u8 = case[0];
        const expected: []const u8 = case[1];

        var buf: [128]u8 = undefined;
        const dest = buf[0..computeSize(input.len)];
        const n = Encoder.encodeToSlice(dest, input);
        try std.testing.expectEqualSlices(u8, expected, dest[0..n]);

        const allocated = try Encoder.encode(std.testing.allocator, input);
        defer std.testing.allocator.free(allocated);
        try std.testing.expectEqualSlices(u8, expected, allocated);
    }
}

test "Encoder tier boundary sizes" {
    // Exhaustively cover sizes around the 4-byte (tier-2) and 16-byte
    // (tier-1) batched-encode boundaries, plus the 0/1/2-byte scalar tail.
    var prng = std.Random.DefaultPrng.init(0xdeadbeef);
    const random = prng.random();

    var size: usize = 0;
    while (size <= 40) : (size += 1) {
        const input = try std.testing.allocator.alloc(u8, size);
        defer std.testing.allocator.free(input);
        random.bytes(input);

        const dest = try std.testing.allocator.alloc(u8, computeSize(size));
        defer std.testing.allocator.free(dest);

        const n = Encoder.encodeToSlice(dest, input);
        try std.testing.expectEqual(dest.len, n);

        // Round-trip through the decoder as an independent correctness
        // check on the encoded bytes, rather than re-deriving expected
        // base64 text by hand for every size.
        const decoded = try Decoder.decode(std.testing.allocator, dest, .reject);
        defer std.testing.allocator.free(decoded);
        try std.testing.expectEqualSlices(u8, input, decoded);
    }
}

test "Encoder tolerates oversized dest" {
    // Mirrors std.base64's "base64 padding dest overflow" test: encoding
    // into a larger-than-required buffer must still produce the correct
    // output in the leading bytes, without requiring an exact-size match.
    const input = "foo";
    const required = computeSize(input.len);

    var exact: [128]u8 = undefined;
    @memset(&exact, 0);
    const exact_n = Encoder.encodeToSlice(exact[0..required], input);

    var oversized: [128]u8 = undefined;
    @memset(&oversized, 0xAA);
    const oversized_n = Encoder.encodeToSlice(&oversized, input);

    try std.testing.expectEqual(exact_n, oversized_n);
    try std.testing.expectEqualSlices(u8, exact[0..exact_n], oversized[0..oversized_n]);
}

test "Decoder known vectors" {
    const cases = .{
        .{ "", "" },
        .{ "f", "Zg==" },
        .{ "fo", "Zm8=" },
        .{ "foo", "Zm9v" },
        .{ "foob", "Zm9vYg==" },
        .{ "fooba", "Zm9vYmE=" },
        .{ "foobar", "Zm9vYmFy" },
        // Long enough to cross the tier-1 (16 source bytes) batched
        // decode boundary.
        .{ "foobarfoobarfoobarfoobarfoobarfoo", "Zm9vYmFyZm9vYmFyZm9vYmFyZm9vYmFyZm9vYmFyZm9v" },
    };

    inline for (cases) |case| {
        const expected: []const u8 = case[0];
        const encoded: []const u8 = case[1];

        var buf: [128]u8 = undefined;
        const dest = buf[0..try computeSizeForSlice(encoded)];
        const n = try Decoder.decodeToSlice(dest, encoded, .reject);
        try std.testing.expectEqualSlices(u8, expected, dest[0..n]);

        const allocated = try Decoder.decode(std.testing.allocator, encoded, .reject);
        defer std.testing.allocator.free(allocated);
        try std.testing.expectEqualSlices(u8, expected, allocated);
    }
}

test "Decoder tier boundary sizes" {
    // Exhaustively cover sizes around the 4-byte (tier-2) and 16-byte
    // (tier-1) batched-decode boundaries by round-tripping through the
    // encoder, rather than hand-deriving expected base64 text per size.
    var prng = std.Random.DefaultPrng.init(0xdeadbeef);
    const random = prng.random();

    var size: usize = 0;
    while (size <= 40) : (size += 1) {
        const input = try std.testing.allocator.alloc(u8, size);
        defer std.testing.allocator.free(input);
        random.bytes(input);

        const encoded = try std.testing.allocator.alloc(u8, computeSize(size));
        defer std.testing.allocator.free(encoded);
        _ = Encoder.encodeToSlice(encoded, input);

        const decoded = try std.testing.allocator.alloc(u8, try computeSizeForSlice(encoded));
        defer std.testing.allocator.free(decoded);
        const n = try Decoder.decodeToSlice(decoded, encoded, .reject);

        try std.testing.expectEqual(input.len, n);
        try std.testing.expectEqualSlices(u8, input, decoded[0..n]);
    }
}

fn decodeErrOnly(encoded: []const u8) !usize {
    var buf: [128]u8 = undefined;
    const size = try computeSizeForSlice(encoded);
    return Decoder.decodeToSlice(buf[0..size], encoded, .reject);
}

test "Decoder rejects invalid length/padding" {
    // Length not a multiple of 4, folds into InvalidPadding for this
    // padded-only encoding, rather than a distinct InvalidLength.
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("A"));
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("AA"));
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("AAA"));

    // Malformed or misplaced padding within an otherwise valid length.
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("AA=A"));
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("A/=="));
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("A==="));
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("===="));
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("Zm9vYmFyZm9vYmFyAA=A"));
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("Zm9vYmFyZm9vYmFyA/=="));
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("Zm9vYmFyZm9vYmFyA==="));
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("Zm9vYmFyZm9vAA=A"));
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("Zm9vYmFyZm9vA/=="));
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("Zm9vYmFyZm9vA==="));

    // Canonical positive control for the 1-remaining-byte case: "Zg==" is
    // the valid encoding of "f" (low bits of the last significant char
    // are 0).
    try std.testing.expectEqual(@as(usize, 1), try decodeErrOnly("Zg=="));

    // Canonical positive control for the 2-remaining-byte case: "Zm8=" is
    // the valid encoding of "fo".
    try std.testing.expectEqual(@as(usize, 2), try decodeErrOnly("Zm8="));

    // "A/==" has v2=63 (0x3F) -> discarded bits nonzero while
    // bytes_to_write==1 -> must be rejected as non-canonical under
    // the default `.reject` policy.
    try std.testing.expectError(error.InvalidPadding, decodeErrOnly("A/=="));
}

test "Decoder NonCanonicalPolicy.allow accepts non-canonical padding" {
    var buf: [128]u8 = undefined;
    const size = try computeSizeForSlice("A/==");
    const n = try Decoder.decodeToSlice(buf[0..size], "A/==", .allow);

    // 'A'(0)<<18 | '/'(63)<<12 = 0x3F000; byte1 = (0x3F000 >> 16) & 0xFF = 0x03.
    // Non-canonical: the discarded low bits (block & 0xFFFF = 0xF000) are
    // nonzero, which .reject would catch but .allow intentionally ignores.
    try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqualSlices(u8, &[_]u8{0x03}, buf[0..n]);
}

test "Decoder rejects invalid characters" {
    try std.testing.expectError(error.InvalidCharacter, decodeErrOnly("A..A"));
    try std.testing.expectError(error.InvalidCharacter, decodeErrOnly("Zm9vYmFyZm9vYmFyA..A"));
    try std.testing.expectError(error.InvalidCharacter, decodeErrOnly("A..AZm9vYmFyZm9vYmFy"));
    // Invalid character inside the batched (non-last) blocks of a longer
    // input, to exercise the tier-1/tier-2 loops' error paths specifically
    // rather than only the scalar last-block path.
    try std.testing.expectError(error.InvalidCharacter, decodeErrOnly("Zm9v!mFyZm9vYmFyZm9vYmFyZm9vYmFyZm9vYmFyZm9v"));
}

test "Decoder rejects undersized dest (NoSpaceLeft)" {
    const cases = [_][]const u8{ "Zg==", "Zm8=", "Zm9v", "Zm9vYg==" };

    for (cases) |encoded| {
        const required = try computeSizeForSlice(encoded);
        if (required == 0) continue;

        var buf: [128]u8 = undefined;
        const dest = buf[0 .. required - 1];
        try std.testing.expectError(error.NoSpaceLeft, Decoder.decodeToSlice(dest, encoded, .reject));
    }
}
