//! Base64 encoding and decoding, as specified by
//! [RFC 4648](https://datatracker.ietf.org/doc/html/rfc4648).
//!
//! Base64 works in 3-byte (24-bit) blocks, split into 4 groups of 6 bits.
//! Each 6-bit group maps to one character of a 64-character alphabet:
//!   0-25  -> uppercase ASCII letters [A-Z]
//!   26-51 -> lowercase ASCII letters [a-z]
//!   52-61 -> digits [0-9]
//!   62-63 -> `+` and `/`
//!
//! When the input length isn't a multiple of 3 bytes, the output is padded
//! with `=` so that the encoded length is always a multiple of 4:
//!   * 1 leftover byte  = 2 chars + "=="
//!   * 2 leftover bytes = 3 chars + "="
//!
//! 3 bytes were chosen as the block size because it's the smallest number
//! of bytes whose total bit count (n * 8) is evenly divisible by 6:
//! the smallest n solving `8n % 6 == 0` is n = 3, giving 24 bits -> 4
//! six-bit characters.

const std = @import("std");
const builtin = @import("builtin");

const little_endian_only = builtin.target.cpu.arch.endian() == .little;

const VecIn = @Vector(24, u8);
const Vec16 = @Vector(16, u8);
const Vec32 = @Vector(32, u8);

const Vec16u16 = @Vector(16, u16);

// WARN!: this @bitCast assumes a little-endian target.
//
// `pairs` is laid out as [lo, hi] byte pairs; reinterpreting as u16 lanes produces (hi << 8) | lo
// ONLY on little-endian. On big-endian the same bytes decode as (lo << 8) | hi, silently corrupting
// every extracted 6-bit value, wrong output, no crash, full speed.
//
// This is guarded by a @compileError at container scope, so BE targets fail to build rather than
// miscompile. If BE support is ever needed, the fix is to emit `pair_idx` with swapped [hi, lo]
// pair order under `builtin.target.cpu.arch.endian() == .big` (purely comptime, zero runtime cost),
// and verify it under QEMU, since no BE hardware is expected in CI.
const Vec32u16 = @Vector(32, u16);

const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Converts 6-bit values in the range 0..63 to Base64 ASCII.
inline fn translate(
    comptime N: comptime_int,
    v: @Vector(N, u8),
) @Vector(N, u8) {
    const Vec = @Vector(N, u8);

    const zero: Vec = @splat(0);

    var off: Vec = @splat(65); // 'A'

    // 0..25  => 'A'..'Z'
    // 26..51 => 'a'..'z'
    // 52..61 => '0'..'9'
    // 62     => '+'
    // 63     => '/'
    off +%= @select(
        u8,
        v > @as(Vec, @splat(25)),
        @as(Vec, @splat(6)),
        zero,
    );

    off -%= @select(
        u8,
        v > @as(Vec, @splat(51)),
        @as(Vec, @splat(75)),
        zero,
    );

    off -%= @select(
        u8,
        v == @as(Vec, @splat(62)),
        @as(Vec, @splat(15)),
        zero,
    );

    off -%= @select(
        u8,
        v == @as(Vec, @splat(63)),
        @as(Vec, @splat(12)),
        zero,
    );

    return v +% off;
}

/// Builds the [lo, hi] byte-pair expansion mask.
///
/// Each input triple [a, b, c] becomes:
///
///     [a, a, b, a, c, b, c, c]
///
/// The resulting byte pairs are:
///
///     [a, a] [b, a] [c, b] [c, c]
///
/// The caller supplies lane_pos because the 32-byte path has a four-byte
/// gap between its two overlapping loads, while the 16-byte path does not.
fn makePairIdx(
    comptime groups: usize,
    comptime lane_pos: fn (usize) usize,
) [groups * 8]i32 {
    const orig = [_]u6{
        0, 0, 1, 0,
        2, 1, 2, 2,
    };

    var idx: [groups * 8]i32 = undefined;

    for (0..groups) |g| {
        for (0..8) |k| {
            const source =
                3 * g + @as(usize, orig[k]);

            idx[8 * g + k] = @intCast(lane_pos(source));
        }
    }

    return idx;
}

const pair_idx = makePairIdx(
    8,
    struct {
        fn f(s: usize) usize {
            // Source bytes 0..11 are in chunk positions 0..11.
            // Source bytes 12..23 are in chunk positions 16..27.
            return if (s < 12) s else s + 4;
        }
    }.f,
);

const pair_idx16 = makePairIdx(
    4,
    struct {
        fn f(s: usize) usize {
            return s;
        }
    }.f,
);

/// Builds a shuffle mask that packs bytes from two overlapping loads.
///
/// Positive indices select from the first shuffle operand.
/// Bitwise-complemented indices select from the second operand.
fn makeLoadIdx(
    comptime output_width: usize,
    comptime a_count: usize,
    comptime b_output_offset: usize,
    comptime b_source_offset: usize,
    comptime b_count: usize,
) [output_width]i32 {
    var idx: [output_width]i32 = undefined;

    // Padding is unused by the subsequent masks. Zero is a valid value.
    for (0..output_width) |j| {
        idx[j] = 0;
    }

    for (0..a_count) |j| {
        idx[j] = @intCast(j);
    }

    for (0..b_count) |j| {
        const source_index: i32 =
            @intCast(b_source_offset + j);

        idx[b_output_offset + j] = ~source_index;
    }

    return idx;
}

// a = source[i..i+16]
// b = source[i+8..i+24]
//
// Result:
//   chunk[0..12]  = source[0..12]
//   chunk[16..28] = source[12..24]
const load_idx = makeLoadIdx(
    32,
    12,
    16,
    4,
    12,
);

// a = source[i..i+8]
// b = source[i+4..i+12]
//
// Result:
//   chunk[0..8] = source[0..8]
//   chunk[8..12] = source[8..12]
const load_idx16 = makeLoadIdx(
    16,
    8,
    8,
    4,
    4,
);

fn makeShifts(
    comptime lanes: usize,
    comptime quad: [4]u4,
) @Vector(lanes, u16) {
    comptime {
        if (lanes % 4 != 0) {
            @compileError("makeShifts: lane count must be divisible by 4");
        }
    }

    const Vec = @Vector(lanes, u16);
    var result: Vec = undefined;

    for (0..lanes / 4) |group| {
        for (0..4) |lane| {
            result[4 * group + lane] = @as(u16, quad[lane]);
        }
    }

    return result;
}

const shifts = makeShifts(32, .{
    10, 4, 6, 0,
});

const shifts16 = makeShifts(16, .{
    2, 4, 6, 0,
});

inline fn encodeChunk(source: []const u8, i: usize) Vec32 {
    comptime {
        if (!little_endian_only) {
            @compileError(
                "encodeChunk: @bitCast(pair bytes -> u16) requires little-endian",
            );
        }
    }

    const a: Vec16 =
        source[i..][0..16].*;

    const b: Vec16 =
        source[i + 8 ..][0..16].*;

    const chunk: Vec32 =
        @shuffle(u8, a, b, load_idx);

    // Each pair is laid out as [lo, hi].
    // On little-endian targets, @bitCast interprets this as:
    //
    //     w = (hi << 8) | lo
    //
    const pairs: @Vector(64, u8) =
        @shuffle(u8, chunk, undefined, pair_idx);

    const w: Vec32u16 =
        @bitCast(pairs);

    const v: Vec32 =
        @truncate(
            (w >> shifts) &
                @as(Vec32u16, @splat(0x3f)),
        );

    return translate(32, v);
}

inline fn encodeChunk16(source: []const u8, i: usize) Vec16 {
    comptime {
        if (!little_endian_only) {
            @compileError(
                "encodeChunk16: @bitCast(pair bytes -> u16) requires little-endian",
            );
        }
    }

    const a: @Vector(8, u8) =
        source[i..][0..8].*;

    const b: @Vector(8, u8) =
        source[i + 4 ..][0..8].*;

    const chunk: Vec16 =
        @shuffle(u8, a, b, load_idx16);

    // pair_idx16 only references positions 0..11, so the second
    // shuffle operand is not logically used. Keep it defined anyway.
    const zero: Vec16 = @splat(0);

    const pairs: @Vector(32, u8) =
        @shuffle(u8, chunk, zero, pair_idx16);

    const w: Vec16u16 =
        @bitCast(pairs);

    const v: Vec16 =
        @truncate(
            (w >> shifts16) &
                @as(Vec16u16, @splat(0x3f)),
        );

    return translate(16, v);
}

/// Returns the padded Base64 output size.
pub fn computeSize(source_len: usize) usize {
    const groups = source_len / 3 + @intFromBool(source_len % 3 != 0);
    return groups * 4; // cannot overflow: groups <= ceil(maxInt/3)
}

/// Returns the exact decoded size for padded Base64 slice.
///
/// This validates:
/// - length is a multiple of four;
/// - padding is at the end;
/// - there are at most two padding characters.
///
/// It does not validate the Base64 alphabet itself.
pub fn computeSizeForSlice(source: []const u8) !usize {
    const len = source.len;
    if (len % 4 != 0) return error.InvalidPadding;
    if (len == 0) return 0;

    var padding: usize = 0;
    if (source[len - 1] == '=') {
        padding += 1;
    }

    if (len >= 2 and source[len - 2] == '=') {
        padding += 1;
    }

    if (padding > 2) return error.InvalidPadding;

    // Padding is only allowed in the final padding positions.
    const non_padding_len = len - padding;
    for (source[0..non_padding_len]) |byte| {
        if (byte == '=') {
            return error.InvalidPadding;
        }
    }

    // A single padding byte must be the final byte only.
    if (padding == 1 and source[len - 2] == '=') {
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
    /// Encodes `source` into the caller-provided buffer `dest`, returning
    /// the number of bytes written.
    ///
    /// `dest` must be at least `computeSize(source.len)` bytes long, otherwise
    /// `error.NoSpaceLeft` is returned. Use `computeSize` to compute the
    /// required length ahead of time. `dest` may be larger than required;
    /// callers may reuse an oversized scratch buffer across calls.
    ///
    /// Encodes source into dest and returns the number of bytes written.
    pub fn encodeToSlice(dest: []u8, source: []const u8) usize {
        const dest_size = computeSize(source.len);
        std.debug.assert(dest.len >= dest_size);
        const source_len = source.len;
        if (source_len == 0) {
            return 0;
        }

        var source_index: usize = 0;
        var dest_index: usize = 0;

        // Fast scalar path for inputs too short for the vector loop;
        // skips hoisting the ymm constants entirely.
        if (source_len >= 24) {
            @branchHint(.likely);
            while (source_index + 24 <= source_len) : ({
                source_index += 24;
                dest_index += 32;
            }) {
                dest[dest_index..][0..32].* = encodeChunk(source, source_index);
            }
        }

        // 12 bytes -> 16 chars, half-width version of encodeChunk.
        // No lane crossing at all: a single 16-byte shuffle does the
        // whole pair expansion.
        // if (source_len - source_index >= 12) {
        //     const c: Vec16 = source[source_index..][0..16].*; // reads 16, uses 12
        //     // pair_idx16: for g in 0..4, emit [3g, 3g, 3g+1, 3g, 3g+1,
        //     //            3g+2, 3g+2, 3g+2] — same orig table, 4 groups
        //     const pairs: @Vector(32, u8) = @shuffle(u8, c, undefined, pair_idx16);
        //     const w: @Vector(16, u16) = @bitCast(pairs);
        //     const v: Vec16 = @truncate((w >> shifts16) & @as(@Vector(16, u16), @splat(0x3f)));
        //
        //     // same compare/select ladder as encodeChunk, at Vec16 width
        //     dest[dest_index..][0..16].* = v +% off16;
        //     source_index += 12;
        //     dest_index += 16;
        // }

        // xmm half-chunk: 12 bytes -> 16 chars for the 12..23 byte band.
        // Runs at most once (remaining < 24 here); the @Vector(16) ops
        // stay in xmm registers with no lane-crossing at all.
        if (source_index + 12 <= source_len) {
            dest[dest_index..][0..16].* = encodeChunk16(source, source_index);
            source_index += 12;
            dest_index += 16;
        }

        // Process exactly 3 source bytes as 4 Base64 characters
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

                dest[dest_index + 0] =
                    alphabet[first >> 2];

                dest[dest_index + 1] =
                    alphabet[(first & 0x03) << 4];

                dest[dest_index + 2] = '=';
                dest[dest_index + 3] = '=';
            },

            2 => {
                const first = source[source_index];
                const second = source[source_index + 1];

                dest[dest_index + 0] =
                    alphabet[first >> 2];

                dest[dest_index + 1] =
                    alphabet[
                        ((first & 0x03) << 4) | (second >> 4)
                    ];

                dest[dest_index + 2] =
                    alphabet[(second & 0x0f) << 2];

                dest[dest_index + 3] = '=';
            },

            else => unreachable,
        }

        return dest_size;
    }

    /// Encodes `source` into a newly allocated buffer using `allocator`.
    ///
    /// The caller owns the returned slice and is responsible for freeing it.
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
