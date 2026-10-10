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
//!
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

const simd = @import("simd.zig");

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

/// Computed once, at comptime, from the actual compilation target, not re-derived here.
/// See simd_tier.zig; build.zig calls the same function against the resolved build target
/// so the two can't drift apart.
const tier: simd.Tier = .fromCpu(builtin.cpu);

const asm_ok =
    tier == .avx2 and
    builtin.cpu.arch == .x86_64 and
    builtin.zig_backend != .stage2_x86_64;

/// One 32-byte input/output chunk.
const Chunk = simd.u8x32;

/// One 16-byte input/output chunk.
const Chunk16 = simd.u8x16;

/// Sixteen little-endian u16 words, each containing one [lo, hi] pair.
const Pairs16 = simd.u16x16;

/// Thirty-two little-endian u16 words, each containing one [lo, hi] pair.
const Pairs = simd.u16x32;

pub const stream_in_len: usize = 12 << 10;
pub const small_in_len: usize = 4 << 10;

comptime {
    if (stream_in_len % 24 != 0) @compileError("stream_in_len must be a multiple of 24");
}

/// Expands the 32-lane `chunk` (24 source bytes, 8 groups of 3) into
/// [low, high] sextet-byte pairs for encodeChunk()'s u16 bitcast.
const pair_indices_8_groups = simd.expandTriplePairs(8);

/// Expands the 16-lane `chunk` (12 source bytes, 4 groups of 3) into
/// [low, high] sextet-byte pairs for encodeChunk16()'s u16 bitcast.
const pair_indices_4_groups = simd.expandTriplePairs(4);

/// Per-lane shift amounts for extracting 4 sextets from 2 packed `u16` pairs
/// (8 source-bytes => 4 base64 characters), repeated across all 8 groups of a
/// 32-lane chunk. Each `u16` lane holds [lo=byte0, hi=byte1] of a pair;
/// See `pair_indices_8_groups` for how `encodeChunk` builds those pairs.
///
/// Shift set `.{10, 4, 6, 0}` pulls out, in output order:
/// 10 = top 6 bits o byte0 (sextet 0)
/// 4  = low 2 bits o byte0 + top 4 bits of byte1 (sextet 1, via 16-bit lane straddling both bytes)
/// 6  = mid 6 bits spanning byte1/byte2 boundary (sextet 2)
/// 0  = low 6 bits of the high byte (sextet 3)
const shifts_32_lanes = simd.laneShifts(32, .{ 10, 4, 6, 0 });

/// Same shift pattern as `shifts_32_lanes`, tiled across 4 groups instead
/// of 8 — used by `encodeChunk16` for the 12-byte/16-char remainder chunk.
const shifts_16_lanes = simd.laneShifts(16, .{ 10, 4, 6, 0 });

// Combines two overlapping 16-byte loads into a 32-byte chunk.
//
// A contributes source[i .. i+12] at chunk[0 .. 12].
// B contributes source[i+12 .. i+24] at chunk[16 .. 28].
//
// The remaining lanes are padding.
const load_indices_32_lanes = simd.loadIndices(
    32,
    .{ .src = 0, .dst = 0, .len = 12 },
    .{ .src = 4, .dst = 16, .len = 12 },
);

// Combines two overlapping 8-byte loads into a 16-byte chunk.
//
// A contributes source[i .. i+8] at chunk[0 .. 8].
// B contributes source[i+8 .. i+12] at chunk[8 .. 12].
//
// The remaining lanes are padding.
const load_indices_16_lanes = simd.loadIndices(
    16,
    .{ .src = 0, .dst = 0, .len = 8 },
    .{ .src = 4, .dst = 8, .len = 4 },
);

fn EncoderImpl(comptime alphabet_chars: [64]u8, comptime pad_char: ?u8) type {
    return struct {
        const Self = @This();

        const alphabet = alphabet_chars;
        const small_out_len: usize = calcSize(&.{}, small_in_len);

        pub const stream_out_len: usize = blk: {
            // stream chunks always take the padded size of stream_in_len; since
            // stream_in_len % 3 == 0 the padded and unpadded sizes agree except
            // on the final short chunk, which encodeToSlice sizes itself.
            break :blk @divTrunc(stream_in_len + 2, 3) * 4;
        };

        /// Maps two Base64 sextets to two packed ASCII bytes.
        ///
        /// Each entry is stored as a little-endian u16 so the bytes are written in
        /// alphabet order by `writeInt(u16, ..., .little)`.
        const pair_lut: [4096]u16 = blk: {
            @setEvalBranchQuota(4 << 10);

            var t: [4096]u16 = undefined;
            for (&t, 0..) |*e, i| {
                e.* = @as(u16, alphabet[i >> 6]) |
                    (@as(u16, alphabet[i & 0x3f]) << 8);
            }

            break :blk t;
        };

        /// Translates 6-bit sextet to ASCII via the offset ladder.
        ///
        /// Caller must ensure `source` is already masked to 6 bits.
        inline fn sextetsToAscii(comptime size: usize, source: @Vector(size, u8)) @Vector(size, u8) {
            const Vec = @Vector(size, u8);

            const upper_offset: Vec = @splat(alphabet[0]);
            const lower_offset: Vec = @splat(alphabet[26] -% 26);
            const digit_offset: Vec = @splat(alphabet[52] -% 52);
            const char62_offset: Vec = @splat(alphabet[62] -% 62);
            const char63_offset: Vec = @splat(alphabet[63] -% 63);

            var offset = upper_offset;

            // 26..51
            offset = @select(
                u8,
                source > @as(Vec, @splat(25)),
                lower_offset,
                offset,
            );

            // 52..63
            offset = @select(
                u8,
                source > @as(Vec, @splat(51)),
                digit_offset,
                offset,
            );

            // 62
            offset = @select(
                u8,
                source == @as(Vec, @splat(62)),
                char62_offset,
                offset,
            );

            // 63
            offset = @select(
                u8,
                source >= @as(Vec, @splat(63)),
                char63_offset,
                offset,
            );

            return source +% offset;
        }

        inline fn writePair12(dest: []u8, index: usize, packed_sextets: u12) void {
            std.mem.writeInt(u16, dest[index..][0..2], pair_lut[packed_sextets], .little);
        }

        inline fn writeQuad24(dest: []u8, i: usize, bits: u24) void {
            writePair12(dest, i, @truncate(bits >> 12));
            writePair12(dest, i + 2, @truncate(bits));
        }

        /// Encodes 24 input bytes starting at `index` into 32 Base64 bytes.
        ///
        /// Uses little-endian pair bit-casts. Big-endian targets are rejected at
        /// compile time.
        fn encodeChunk(source: []const u8, index: usize) simd.u8x32 {
            const a: simd.u8x16 = source[index..][0..16].*;
            const b: simd.u8x16 = source[index + 8 ..][0..16].*;

            const chunk: Chunk = @shuffle(
                u8,
                a,
                b,
                load_indices_32_lanes,
            );

            const pairs: simd.u8x64 = @shuffle(
                u8,
                chunk,
                chunk,
                pair_indices_8_groups,
            );

            const words: Pairs = @bitCast(pairs);
            const sextets: Pairs = (words >> shifts_32_lanes) & @as(Pairs, @splat(0x3f));
            return sextetsToAscii(32, @truncate(sextets));
        }

        /// Encodes 12 input bytes starting at `i` into 16 Base64 bytes.
        ///
        /// Uses little-endian pair bit-casts. Big-endian targets are rejected at
        /// compile time.
        fn encodeChunk16(source: []const u8, i: usize) Chunk16 {
            const a: @Vector(8, u8) = source[i..][0..8].*;
            const b: @Vector(8, u8) = source[i + 4 ..][0..8].*;

            const chunk: Chunk16 = @shuffle(u8, a, b, load_indices_16_lanes);

            const pairs: @Vector(32, u8) = @shuffle(
                u8,
                chunk,
                chunk,
                pair_indices_4_groups,
            );
            const w: Pairs16 = @bitCast(pairs);
            const sextets: Pairs16 = (w >> shifts_16_lanes) & @as(Pairs16, @splat(0x3f));
            return sextetsToAscii(16, @truncate(sextets));
        }

        /// Returns the padded Base64 output size.
        pub fn calcSize(_: *const Self, source_len: usize) usize {
            if (comptime pad_char != null) {
                return @divTrunc(source_len + 2, 3) * 4;
            } else {
                const leftover = source_len % 3;
                return @divTrunc(source_len, 3) * 4 + @divTrunc(leftover * 4 + 2, 3);
            }
        }

        pub fn encodeToSlice(self: *const Self, dest: []u8, source: []const u8) usize {
            if (source.len == 0) {
                return 0;
            }

            const dest_size = self.calcSize(source.len);
            const source_len = source.len;

            std.debug.assert(dest.len >= dest_size);

            var source_index: usize = 0;
            var dest_index: usize = 0;

            switch (tier) {
                .avx2 => {
                    // ymm main loop: 24 bytes for 32 chars.

                    while (source_index + 24 <= source_len) : ({
                        source_index += 24;
                        dest_index += 32;
                    }) {
                        dest[dest_index..][0..32].* = encodeChunk(source, source_index);
                    }

                    // xmm half-chunk: 12 bytes for 16 chars;
                    // runs at most once here (remaining < 24 bytes).
                    if (source_index + 12 <= source_len) {
                        dest[dest_index..][0..16].* = encodeChunk16(source, source_index);
                        source_index += 12;
                        dest_index += 16;
                    }
                },
                .simd128 => {
                    // xmm loop: 12 bytes for 16 chars; bulk path.
                    while (source_index + 12 <= source_len) : ({
                        source_index += 12;
                        dest_index += 16;
                    }) {
                        dest[dest_index..][0..16].* = encodeChunk16(source, source_index);
                    }
                },
                .scalar => {
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
                        writePair12(dest, dest_index + 0, @truncate(hi >> 52));
                        writePair12(dest, dest_index + 2, @truncate(hi >> 40));
                        writePair12(dest, dest_index + 4, @truncate(hi >> 28));
                        writePair12(dest, dest_index + 6, @truncate(hi >> 16));
                        writePair12(dest, dest_index + 8, @truncate(hi >> 4));

                        // boundary pair: hi bits 3..0 ++ mid bits 31..24
                        writePair12(
                            dest,
                            dest_index + 10,
                            @truncate((hi & 0xf) << 8 | (mid >> 24)),
                        );

                        // mid chain: remaining pairs for bytes 8..11
                        writePair12(dest, dest_index + 12, @truncate(mid >> 12));
                        writePair12(dest, dest_index + 14, @truncate(mid));
                    }
                },
            }

            // Full-triple tail: <=3 iterations on .avx2/.simd128,
            // up to 5 on .scalar (u128 loop exits with <= 15 bytes left).
            while (source_index + 3 <= source_len) : ({
                source_index += 3;
                dest_index += 4;
            }) {
                const bits = std.mem.readInt(u24, source[source_index..][0..3], .big);
                writeQuad24(dest, dest_index, bits);
            }

            // Encode the remaining zero, one, or two bytes.
            switch (source_len - source_index) {
                0 => {},

                1 => {
                    const first = source[source_index];
                    dest[dest_index + 0] = alphabet[first >> 2];
                    dest[dest_index + 1] = alphabet[(first & 0x03) << 4];
                    if (comptime pad_char) |p| {
                        dest[dest_index + 2] = p;
                        dest[dest_index + 3] = p;
                    }
                },

                2 => {
                    const first = source[source_index];
                    const second = source[source_index + 1];

                    dest[dest_index + 0] = alphabet[first >> 2];
                    dest[dest_index + 1] = alphabet[((first & 0x03) << 4) | (second >> 4)];
                    dest[dest_index + 2] = alphabet[(second & 0x0f) << 2];

                    if (comptime pad_char) |p| {
                        dest[dest_index + 3] = p;
                    }
                },

                else => unreachable,
            }

            return dest_size;
        }

        pub fn encode(self: *const Self, dest: []u8, source: []const u8) []const u8 {
            const n = self.encodeToSlice(dest, source);
            return dest[0..n];
        }

        /// Encodes `source` and writes the result to `writer` in cache-resident chunks.
        ///
        /// This avoids allocating an output buffer proportional to `source.len`.
        ///
        /// For inputs up to `small_in_len` (4 KiB) encode through a small stack buffer in
        /// a single `writeAll`, it has te same speed as `encodeToSlice` plus one writer
        /// dispatch. For larger inputs stream through a ~16KiB scratch buffer, which ties
        /// `encodeToSlice` up to ~4 MiB and wins ~5–14% beyond that, since the staging
        /// buffer stays cache-resident while a multi-MiB destination does not (measured
        /// on znver2; crossover varies with LLC size).
        ///
        /// Below ~64 B the writer dispatch overhead dominates; use `encodeToSlice` for
        /// small inputs or when a contiguous output buffer is required.
        /// On error, `writer` may contain a prefix of the encoded output.
        pub fn encodeWriter(
            self: *const Self,
            writer: *std.Io.Writer,
            source: []const u8,
        ) std.Io.Writer.Error!void {
            if (source.len <= small_in_len) {
                // ~5.5 KiB frame; stack probing handled by the compiler.
                var small: [small_out_len]u8 = undefined;
                const written = self.encodeToSlice(&small, source);
                return writer.writeAll(small[0..written]);
            }
            return self.encodeWriterChunked(writer, source);
        }

        noinline fn encodeWriterChunked(
            self: *const Self,
            writer: *std.Io.Writer,
            source: []const u8,
        ) std.Io.Writer.Error!void {
            var scratch: [stream_out_len]u8 = undefined;
            var offset: usize = 0;
            while (offset < source.len) {
                const n = @min(stream_in_len, source.len - offset);
                const written = self.encodeToSlice(&scratch, source[offset..][0..n]);
                try writer.writeAll(scratch[0..written]);
                offset += n;
            }
        }

        /// Encodes `source` into a newly allocated buffer.
        ///
        /// The caller owns the returned slice and must free it with `gpa`.
        /// Returns an allocator error if the output buffer cannot be allocated.
        pub fn allocEncode(self: Self, gpa: std.mem.Allocator, source: []const u8) ![]u8 {
            const out = try gpa.alloc(u8, self.calcSize(source.len));
            _ = self.encodeToSlice(out, source);
            // return out directly since n == computeSize(source.len) == out.len always.
            return out;
        }
    };
}

/// One Base64 variant: alphabet + padding + a specialized encoder.
/// Mirrors std.base64.Codecs, minus the decoder side for now.
pub fn Codecs(comptime alphabet: [64]u8, comptime pad: ?u8) type {
    return struct {
        pub const alphabet_chars = alphabet;
        pub const pad_char = pad;

        /// Zero-field struct: `standard.Encoder` is both the type's
        /// namespace and a callable value, matching std's usage
        /// `standard.Encoder.encode(dest, src)`.
        pub const Encoder: EncoderImpl(alphabet, pad) = .{};

        pub const Decoder: type = void;
    };
}

/// The Base64 alphabet defined in
/// [RFC 4648 section 4](https://datatracker.ietf.org/doc/html/rfc4648#section-4).
pub const standard_alphabet_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".*;

/// The URL-safe Base64 alphabet defined in
/// [RFC 4648 section 5](https://datatracker.ietf.org/doc/html/rfc4648#section-5).
pub const url_safe_alphabet_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".*;

pub const standard = Codecs(standard_alphabet_chars, '=');
pub const standard_no_pad = Codecs(standard_alphabet_chars, null);
pub const url_safe = Codecs(url_safe_alphabet_chars, '=');
pub const url_safe_no_pad = Codecs(url_safe_alphabet_chars, null);

const testing = std.testing;

fn testEncode(comptime Codec: type, src: []const u8, expected: []const u8) !void {

    // encode to slice
    {
        const out_len = Codec.Encoder.calcSize(src.len);
        try std.testing.expectEqual(expected.len, out_len);

        var buf: [256]u8 = undefined;
        const n = Codec.Encoder.encodeToSlice(buf[0..out_len], src);
        try testing.expectEqualStrings(expected, buf[0..n]);
    }

    // encode with alloc
    {
        const buf = try Codec.Encoder.allocEncode(testing.allocator, src);
        defer testing.allocator.free(buf);
        try testing.expectEqualStrings(expected, buf);
    }

    // encode to stream writer
    {
        var aw: std.Io.Writer.Allocating = .init(testing.allocator);
        defer aw.deinit();
        try Codec.Encoder.encodeWriter(&aw.writer, src);
        try testing.expectEqualStrings(expected, aw.written());
    }
}

test "Base64Encoder: fixed vectors, standard" {
    try testEncode(standard, "", "");
    try testEncode(standard, "f", "Zg==");
    try testEncode(standard, "fo", "Zm8=");
    try testEncode(standard, "foo", "Zm9v");
    try testEncode(standard, "foob", "Zm9vYg==");
    try testEncode(standard, "fooba", "Zm9vYmE=");
    try testEncode(standard, "foobar", "Zm9vYmFy");

    try testEncode(standard, "\xff\xff\xff", "////");
    try testEncode(standard, "\xfb\xff\xbf", "+/+/");

    try testEncode(standard, "\x00\x10\x83\x10\x51\x87\x20\x92\x8b", "ABCDEFGHIJKL");
}

test "Base64Encoder: fixed vectors, standard_no_pad" {
    try testEncode(standard_no_pad, "", "");
    try testEncode(standard_no_pad, "f", "Zg");
    try testEncode(standard_no_pad, "fo", "Zm8");
    try testEncode(standard_no_pad, "foo", "Zm9v");
    try testEncode(standard_no_pad, "foob", "Zm9vYg");
    try testEncode(standard_no_pad, "fooba", "Zm9vYmE");
    try testEncode(standard_no_pad, "foobar", "Zm9vYmFy");
}

test "Base64Encoder: fixed vectors, url_safe" {
    try testEncode(url_safe, "", "");
    try testEncode(url_safe, "f", "Zg==");
    try testEncode(url_safe, "fo", "Zm8=");
    try testEncode(url_safe, "foo", "Zm9v");
    try testEncode(url_safe, "foob", "Zm9vYg==");
    try testEncode(url_safe, "fooba", "Zm9vYmE=");
    try testEncode(url_safe, "foobar", "Zm9vYmFy");
    try testEncode(url_safe, "\xfb\xff\xbf", "-_-_");
    try testEncode(url_safe, "\xff\xff\xff", "____");
}

test "Base64Encoder: fixed vectors, url_safe_no_pad" {
    try testEncode(url_safe_no_pad, "", "");
    try testEncode(url_safe_no_pad, "f", "Zg");
    try testEncode(url_safe_no_pad, "fo", "Zm8");
    try testEncode(url_safe_no_pad, "foo", "Zm9v");
    try testEncode(url_safe_no_pad, "foob", "Zm9vYg");
    try testEncode(url_safe_no_pad, "fooba", "Zm9vYmE");
    try testEncode(url_safe_no_pad, "foobar", "Zm9vYmFy");
}

test "Base64Encoder: differential vs std.base64, all codecs" {
    const Pair = struct { custom: type, std: std.base64.Codecs };
    const pairs = [_]Pair{
        .{ .custom = standard, .std = std.base64.standard },
        .{ .custom = standard_no_pad, .std = std.base64.standard_no_pad },
        .{ .custom = url_safe, .std = std.base64.url_safe },
        .{ .custom = url_safe_no_pad, .std = std.base64.url_safe_no_pad },
    };

    var prng = std.Random.DefaultPrng.init(0xdeadbeef);
    const rnd = prng.random();

    inline for (pairs) |p| {
        for (0..1024) |len| {
            var src: [1024]u8 = undefined;
            rnd.bytes(src[0..len]);

            var got: [2048]u8 = undefined;
            var want: [2048]u8 = undefined;

            const n = p.custom.Encoder.encodeToSlice(&got, src[0..len]);
            const wn = p.std.Encoder.encode(&want, src[0..len]).len;

            try testing.expectEqual(wn, n);
            try testing.expectEqualSlices(u8, want[0..wn], got[0..n]);
        }
    }
}

test "Base64Encoder: sextet 62/63 boundary coverage" {
    // All sextets = 63 — the old `source > 63` ladder bug (always false,
    // since masked input never exceeds 63) lived exactly here.
    try testEncode(standard, "\xff\xff\xff", "////");
    try testEncode(url_safe, "\xff\xff\xff", "____");

    // Alternating 62/63 within one 4-sextet group; catches an off-by-one
    // between the adjacent top-of-range branches that an all-63 vector
    // can't distinguish.
    try testEncode(standard, "\xfb\xff\xbf", "+/+/");
    try testEncode(url_safe, "\xfb\xff\xbf", "-_-_");

    // All sextets = 0 and = 26/52 pin the other alphabet-offset boundaries
    // (upper/lower/digit transitions in the offset ladder).
    try testEncode(standard, "\x00\x00\x00", "AAAA");

    // the standard alphabet (0-25 = A-Z, 26-51 = a-z, 52-61 = digits, 62 = +, 63 = /):
    // 0x68 0x1a 0x69 = 01101000 00011010 01101001
    try testEncode(standard, "\x68\x1a\x69", "aBpp");
}

test "Base64Encoder: calcSize matches std for both pad modes" {
    for (0..128) |len| {
        try testing.expectEqual(
            std.base64.standard.Encoder.calcSize(len),
            standard.Encoder.calcSize(len),
        );
        try testing.expectEqual(
            std.base64.standard_no_pad.Encoder.calcSize(len),
            standard_no_pad.Encoder.calcSize(len),
        );
    }
}

test "Base64Encoder: chunk and tier boundary lengths" {
    // Exercise every transition point across the avx2/simd128/scalar tiers:
    // 23/24/25 (ymm chunk boundary), 11/12/13 (xmm chunk boundary),
    // 14/15/16/17 (scalar u128 lookahead boundary), 0/1/2/3 (tail switch).
    const lens = [_]usize{
        0,  1,  2,  3,  4,  5,
        11, 12, 13, 14, 15, 16,
        17, 23, 24, 25, 47, 48,
        49,
    };

    var prng = std.Random.DefaultPrng.init(1);
    const rnd = prng.random();
    for (lens) |len| {
        var src: [64]u8 = undefined;
        rnd.bytes(src[0..len]);
        var got: [128]u8 = undefined;
        var want: [128]u8 = undefined;
        const n = standard.Encoder.encodeToSlice(&got, src[0..len]);
        const wn = std.base64.standard.Encoder.encode(&want, src[0..len]).len;
        try testing.expectEqual(wn, n);
        try testing.expectEqualSlices(u8, want[0..wn], got[0..n]);
    }
}

test "Base64Encoder: encodeWriter matches encodeToSlice across small/chunk/multi-chunk" {
    var prng = std.Random.DefaultPrng.init(2);
    const rnd = prng.random();
    // 0, small_in_len-ish, exactly stream_in_len, stream_in_len+1,
    // several multiples of stream_in_len.
    const lens = [_]usize{ 0, 1, 4095, 4096, 4097, 12288, 12289, 24576, 24577 };
    inline for (lens) |len| {
        const src = try testing.allocator.alloc(u8, len);
        defer testing.allocator.free(src);
        rnd.bytes(src);

        const computeSize_placeholder = comptime standard.Encoder.calcSize(len);
        var direct_buf: [computeSize_placeholder]u8 = undefined; // size via calcSize(len)
        const dn = standard.Encoder.encodeToSlice(direct_buf[0..standard.Encoder.calcSize(len)], src);

        var aw = std.Io.Writer.Allocating.init(testing.allocator);
        defer aw.deinit();
        try standard.Encoder.encodeWriter(&aw.writer, src);

        try testing.expectEqualSlices(u8, direct_buf[0..dn], aw.written());
    }
}

test "Base64Encoder: padding, oversized dest buffer unaffected beyond written length" {
    const input = "foo";
    var expect: [128]u8 = undefined;
    @memset(&expect, 0);
    _ = url_safe.Encoder.encodeToSlice(expect[0..url_safe.Encoder.calcSize(input.len)], input);

    var got: [128]u8 = undefined;
    @memset(&got, 0);
    _ = url_safe.Encoder.encodeToSlice(&got, input); // full 128-byte dest, not just calcSize

    try testing.expectEqualSlices(u8, &expect, &got);
}
