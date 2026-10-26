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
pub const tier: simd.Tier = .fromCpu(builtin.cpu);

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

        /// Encoded length of `small_in_len` byte input. Computed at comptime by
        /// calling the inistance method `calcSize` with a dummy `&.{}` receiver.
        ///
        /// NOTE! `&.{}` only coerces to `*const Self` while `Self` has no fields.
        /// If `EncoderImpl` every gains a field, this line fails with a confusing
        /// "expected type '*const Self'" error; switch to a free helper that doesn't
        /// take `self`.
        const small_out_len: usize = calcSize(&.{}, small_in_len);

        comptime {
            if (@typeInfo(Self).@"struct".fields.len != 0) {
                @compileError(
                    "EncoderImpl must stay field-less: small_out_len calls " ++
                        "calcSize(&.{}, ...) at comptime. See the note on small_out_len",
                );
            }
        }

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
        pub fn allocEncode(self: *const Self, gpa: std.mem.Allocator, source: []const u8) ![]u8 {
            const out = try gpa.alloc(u8, self.calcSize(source.len));
            _ = self.encodeToSlice(out, source);
            // return out directly since n == computeSize(source.len) == out.len always.
            return out;
        }
    };
}

const invalid_char: u8 = 0xff;
const invalid_char_tst: u32 = 0xff00_0000;

pub const Error = error{
    InvalidCharacter,
    InvalidPadding,
    NoSpaceLeft,
};

pub fn DecoderImpl(comptime alphabet_chars: [64]u8, comptime pad_char: ?u8) type {
    return struct {
        const Self = @This();

        /// True when `alphabet_chars` is A-Za-z0-9 followed by any two other characters
        /// (standard: `+/`, URL-safe: `-_`). This is the only layout the range compare
        /// ladder in `classify` handle for.
        const simd_alphabet = std.mem.eql(
            u8,
            alphabet_chars[0..62],
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        );

        const simd_width: usize = switch (tier) {
            .avx2 => 32,
            .simd128 => 16,
            .scalar => 0,
        };

        // Built once per (alphabet, pad_char) instantiation, no init() call,
        // no runtime struct fields. Mirrors Base64Decoder.init's two tables.
        const char_to_index: [256]u8 = blk: {
            var t: [256]u8 = .{invalid_char} ** 256;

            for (alphabet_chars, 0..) |char, index| {
                if (t[char] != invalid_char) {
                    @compileError("decoder: duplicate alphabet character.");
                }
                if (pad_char != null and char == pad_char.?) {
                    @compileError("decoder: alphabet character with pad_char");
                }
                t[char] = @intCast(index);
            }

            break :blk t;
        };

        const fast_char_to_index: [4][256]u32 = blk: {
            var t = [_][256]u32{[_]u32{invalid_char_tst} ** 256} ** 4;
            for (alphabet_chars, 0..) |char, index| {
                const char_idx: u32 = @intCast(index);
                t[0][char] = char_idx << 2;
                t[1][char] = (char_idx >> 4) | ((char_idx & 0x0f) << 12);
                t[2][char] = ((char_idx & 0x3) << 22) | ((char_idx & 0x3c) << 6);
                t[3][char] = char_idx << 16;
            }
            break :blk t;
        };

        /// Upper bound on decoded length; exact only when there's no padding
        /// to subtract. `InvalidPadding` if source_len's shape is impossible.
        pub fn calcSizeUpperBound(_: *const Self, source_len: usize) Error!usize {
            var result = source_len / 4 * 3;
            const leftover = source_len % 4;
            if (comptime pad_char != null) {
                if (leftover != 0) return error.InvalidPadding;
            } else {
                if (leftover == 1) return error.InvalidPadding;
                result += leftover * 3 / 4;
            }

            return result;
        }

        /// Exact decoded length for a specific source slice (accounts for trailing pad chars
        /// actually present).
        pub fn calcSizeForSlice(self: *const Self, source: []const u8) Error!usize {
            const source_len = source.len;
            var result = try self.calcSizeUpperBound(source_len);
            if (comptime pad_char) |p| {
                if (source_len >= 1 and source[source_len - 1] == p) result -= 1;
                if (source_len >= 2 and source[source_len - 2] == p) result -= 1;
            }
            return result;
        }

        // passing *usize through a call can force the
        // indices to spill to the stack on every iteration.
        inline fn simdLoop(
            comptime N: comptime_int,
            dest: []u8,
            source: []const u8,
            src_index: *usize,
            dest_index: *usize,
        ) void {
            const out_n = N / 4 * 3; // 24 for AVX2, 12 for SIMD128
            while (src_index.* + N < source.len and dest_index.* + out_n <= dest.len) {
                const chars: @Vector(N, u8) = source[src_index.*..][0..N].*;
                const sextets = classify(N, chars);
                if (!allValid(N, sextets)) break; // no-op; resumes to scalar
                dest[dest_index.*..][0..out_n].* = pack(N, sextets);
                src_index.* += N;
                dest_index.* += out_n;
            }
        }

        // Out of line so the ymm work (and its vzeroupper) only runs for inputs
        // long enough to reach a full SIMD block.
        noinline fn simdLoopOutline(
            comptime N: comptime_int,
            dest: []u8,
            source: []const u8,
            src_index: *usize,
            dest_index: *usize,
        ) void {
            var si = src_index.*;
            var di = dest_index.*;
            simdLoop(N, dest, source, &si, &di);
            src_index.* = si;
            dest_index.* = di;
        }

        /// Destination length must be exactly as per calcSizeForSlice(source).
        pub fn decode(self: *const Self, dest: []u8, source: []const u8) Error!void {
            return self.decodeWith(simd_width, dest, source);
        }

        fn decodeWith(
            _: *const Self,
            comptime simd_n: usize,
            dest: []u8,
            source: []const u8,
        ) Error!void {
            if (pad_char != null and source.len % 4 != 0) return error.InvalidPadding;

            var dest_idx: usize = 0;
            var fast_src_idx: usize = 0;
            var acc: u32 = 0;
            var acc_len: u32 = 0;
            var leftover_idx: ?usize = null;

            if (tier != .scalar and comptime simd_n > 0 and simd_alphabet) {
                if (source.len > simd_n) simdLoopOutline(simd_n, dest, source, &fast_src_idx, &dest_idx);
            }

            // 16-byte fast loop: 4 lanes of fast_char_to_index OR'd together,
            // one u128 little-endian store per 16 source chars, 12 dest bytes.
            while (fast_src_idx + 16 < source.len and dest_idx + 15 < dest.len) : ({
                fast_src_idx += 16;
                dest_idx += 12;
            }) {
                var bits: u128 = 0;
                var any_invalid: u32 = 0;

                inline for (0..4) |i| {
                    var new_bits: u128 = fast_char_to_index[0][source[fast_src_idx + i * 4]];
                    new_bits |= fast_char_to_index[1][source[fast_src_idx + 1 + i * 4]];
                    new_bits |= fast_char_to_index[2][source[fast_src_idx + 2 + i * 4]];
                    new_bits |= fast_char_to_index[3][source[fast_src_idx + 3 + i * 4]];

                    // Accumulate the sentinel check here, pre-shift, where all four lanes
                    // sentinel bits stil occupy the same [24,31] range, this is collision-free,
                    // unilike check bits after the shift below.
                    any_invalid |= @truncate(new_bits & invalid_char_tst);
                    bits |= (new_bits << (24 * i));
                }

                if (any_invalid != 0) return error.InvalidCharacter;
                std.mem.writeInt(u128, dest[dest_idx..][0..16], bits, .little);
            }

            // 4-byte loop: same table trick, one lane group at a time
            while (fast_src_idx + 4 < source.len and dest_idx + 3 < dest.len) : ({
                fast_src_idx += 4;
                dest_idx += 3;
            }) {
                var bits = fast_char_to_index[0][source[fast_src_idx]];

                bits |= fast_char_to_index[1][source[fast_src_idx + 1]];
                bits |= fast_char_to_index[2][source[fast_src_idx + 2]];
                bits |= fast_char_to_index[3][source[fast_src_idx + 3]];
                if ((bits & invalid_char_tst) != 0) return error.InvalidCharacter;
                std.mem.writeInt(u32, dest[dest_idx..][0..4], bits, .little);
            }

            // Scalar tail: bit accumulator, 6 bits in / 8 bits out.
            const remaining = source[fast_src_idx..];
            for (remaining, fast_src_idx..) |c, src_idx| {
                const d = char_to_index[c];
                if (d == invalid_char) {
                    if (pad_char == null or c != pad_char.?) return error.InvalidCharacter;
                    leftover_idx = src_idx;
                    break;
                }
                acc = ((acc << 6) | d) & 0xfff;
                acc_len += 6;
                if (acc_len >= 8) {
                    acc_len -= 8;
                    dest[dest_idx] = @truncate(acc >> @as(u5, @intCast(acc_len)));
                    dest_idx += 1;
                }
            }

            if (acc_len > 4 or (acc & ((@as(u32, 1) << @as(u5, @intCast(acc_len))) - 1)) != 0) {
                return error.InvalidPadding;
            }

            if (leftover_idx == null) return;
            const leftover = source[leftover_idx.?..];
            if (comptime pad_char) |p| {
                const padding_len = acc_len / 2;
                var padding_chars: usize = 0;
                for (leftover) |c| {
                    if (c != p) {
                        // compares the raw byte against the sentinel value directly
                        return if (c == invalid_char) error.InvalidCharacter else error.InvalidPadding;
                    }
                    padding_chars += 1;
                }

                if (padding_chars != padding_len) return error.InvalidPadding;
            }
        }

        /// Maps each ASCII lane to its sextet, or 0xff if it isn't an alphabet char.
        inline fn classify(comptime N: comptime_int, v: @Vector(N, u8)) @Vector(N, u8) {
            if (comptime !simd_alphabet) {
                @compileError(
                    "classify: unsupported alphabet layout. The SIMD classifier " ++
                        "requires A-Za-z0-9 as the first 62 characters, followed by " ++
                        "any two other characters. Custom alphabets must use the " ++
                        "scalar decode path (see `decode_tier`).",
                );
            }
            const V = @Vector(N, u8);
            const upper = (v -% @as(V, @splat('A'))) <= @as(V, @splat('Z' - 'A'));
            const lower = (v -% @as(V, @splat('a'))) <= @as(V, @splat('z' - 'a'));
            const digit = (v -% @as(V, @splat('0'))) <= @as(V, @splat('9' - '0'));
            const is62 = v == @as(V, @splat(alphabet_chars[62]));
            const is63 = v == @as(V, @splat(alphabet_chars[63]));

            var out: V = @splat(0xff);
            out = @select(u8, upper, v -% @as(V, @splat('A')), out);
            out = @select(u8, lower, (v -% @as(V, @splat('a'))) +% @as(V, @splat(26)), out);
            out = @select(u8, digit, (v -% @as(V, @splat('0'))) +% @as(V, @splat(52)), out);
            out = @select(u8, is62, @as(V, @splat(62)), out);
            out = @select(u8, is63, @as(V, @splat(63)), out);
            return out;
        }

        inline fn pack(comptime N: comptime_int, s: @Vector(N, u8)) @Vector(N / 4 * 3, u8) {
            const L = N / 4;
            const W = @Vector(L, u32);
            const x: W = @bitCast(s);
            const m: W = @splat(0x3f);

            const t = ((x & m) << @as(W, @splat(18))) |
                (((x >> @as(W, @splat(8))) & m) << @as(W, @splat(12))) |
                (((x >> @as(W, @splat(16))) & m) << @as(W, @splat(6))) |
                ((x >> @as(W, @splat(24))) & m);

            // Little-endian lane bytes are [b0, b1, b2, 0]. Emit b2, b1, b0.
            const mask = comptime blk: {
                var idx: [L * 3]i32 = undefined;
                for (0..L) |K| {
                    idx[3 * K + 0] = @intCast(4 * K + 2);
                    idx[3 * K + 1] = @intCast(4 * K + 1);
                    idx[3 * K + 2] = @intCast(4 * K + 0);
                }
                break :blk idx;
            };

            const bytes: @Vector(N, u8) = @bitCast(t);
            return @shuffle(u8, bytes, undefined, mask);
        }

        /// True when every lane of `sextets` came from a valid alphabet character.
        inline fn allValid(comptime N: comptime_int, sextets: @Vector(N, u8)) bool {
            return (@reduce(.Or, sextets) >> 6) == 0;
        }
    };
}

test "DecoderImpl: every decode path agrees with std on clean and corrupted input" {
    var prng = std.Random.DefaultPrng.init(0xdec0de);
    const rng = prng.random();

    inline for (.{
        .{ standard, std.base64.standard },
        .{ standard_no_pad, std.base64.standard_no_pad },
        .{ url_safe, std.base64.url_safe },
        .{ url_safe_no_pad, std.base64.url_safe_no_pad },
    }) |pair| {
        const mine, const theirs = pair;
        for (0..300) |len| {
            var raw: [300]u8 = undefined;
            rng.bytes(raw[0..len]);
            var enc_buf: [512]u8 = undefined;
            const enc = theirs.Encoder.encode(&enc_buf, raw[0..len]);

            for (0..4) |round| {
                if (round != 0 and enc.len != 0) enc_buf[rng.uintLessThan(usize, enc.len)] = rng.int(u8);

                const sz = theirs.Decoder.calcSizeForSlice(enc) catch {
                    try testing.expectError(error.InvalidPadding, mine.Decoder.calcSizeForSlice(enc));
                    continue;
                };
                var want_buf: [512]u8 = undefined;
                var got_buf: [512]u8 = undefined;
                const want: ?anyerror = if (theirs.Decoder.decode(want_buf[0..sz], enc)) |_| null else |e| e;

                inline for (.{ 0, 16, 32 }) |n| {
                    const r = mine.Decoder.decodeWith(n, got_buf[0..sz], enc);
                    if (want) |e| {
                        try testing.expectError(e, r);
                    } else {
                        try r;
                        try testing.expectEqualSlices(u8, want_buf[0..sz], got_buf[0..sz]);
                    }
                }
            }
        }
    }
}

test "DecoderImpl: classify matches char_to_index for every byte at every lane position" {
    inline for (.{ std.base64.standard_alphabet_chars, std.base64.url_safe_alphabet_chars }) |chars| {
        const D = DecoderImpl(chars, '=');
        inline for (.{ 16, 32 }) |N| {
            for (0..256) |base| {
                var in: [N]u8 = undefined;
                for (&in, 0..) |*b, i| b.* = @truncate(base + i);
                const out: [N]u8 = D.classify(N, in);
                for (in, out) |c, got| try testing.expectEqual(D.char_to_index[c], got);
                const ok = for (in) |c| {
                    if (D.char_to_index[c] == invalid_char) break false;
                } else true;

                try testing.expectEqual(ok, D.allValid(N, out));
            }
        }
    }
}

test "DecoderImpl compiles and decodes correctly with non-standard alphabet layout on non-scalar tier" {
    // Deliberately NOT the A-Za-z0-9 prefix order simd_alphabet expects
    // this permutes the first four chars so classify()/simd_alphabet must be false,
    // forcing the scalar path even when `tier` would otherwise pick avx2/simd128.
    const weird_alphabet = "BADCEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".*;

    const D = DecoderImpl(weird_alphabet, '=');

    // This instantiation must compile regardless of what `tier` resolves to
    // on the test target, proves the SIMD block isn't force-instantiated.
    try comptime std.testing.expect(!D.simd_alphabet);

    // Round-trip through std's encoder (built from the same custom alphabet)
    // to confirm the scalar fallback actually decodes correctly, not just compiles.
    const decoded_expected = "foobar";

    const std_encoder = comptime std.base64.Base64Encoder.init(weird_alphabet, '=');
    const encoded_len = comptime std_encoder.calcSize(decoded_expected.len);
    var encoded: [encoded_len]u8 = undefined;
    _ = std_encoder.encode(&encoded, decoded_expected);

    var dest: [decoded_expected.len]u8 = undefined;
    const decoder: D = .{};
    try decoder.decode(&dest, &encoded);
    try std.testing.expectEqualStrings(decoded_expected, &dest);
}

pub fn DecoderWithIgnoreImpl(comptime alphabet_chars: [64]u8, comptime pad_char: ?u8) type {
    const Decoder = DecoderImpl(alphabet_chars, pad_char);

    return struct {
        const Self = @This();

        const char_to_index = Decoder.char_to_index;

        // char_is_ignored is a runtime known, ignore_chars is caller-supplied per call,
        char_is_ignored: [256]bool,

        pub fn init(ignore_chars: []const u8) Self {
            var self: Self = .{ .char_is_ignored = [_]bool{false} ** 256 };

            for (ignore_chars) |c| {
                std.debug.assert(char_to_index[c] == invalid_char);
                std.debug.assert(!self.char_is_ignored[c]);
                std.debug.assert(pad_char == null or pad_char.? != c);

                self.char_is_ignored[c] = true;
            }

            return self;
        }

        /// Maximum possible decoded size, actual length may be less if the input
        /// includes padding or ignored characters.
        pub fn calcSizeUpperBound(_: *const Self, source_len: usize) usize {
            var size = source_len / 4 * 3;
            if (comptime pad_char == null) {
                const leftover = source_len % 4;
                size += leftover * 3 / 4;
            }
            return size;
        }

        pub fn decode(self: *const Self, dest: []u8, source: []const u8) Error!usize {
            var acc: u32 = 0;
            var acc_len: u32 = 0;
            var dest_idx: usize = 0;
            var leftover_idx: ?usize = null;

            for (source, 0..) |c, src_idx| {
                if (self.char_is_ignored[c]) continue;

                const d = char_to_index[c];
                if (d == invalid_char) {
                    if (pad_char == null or c != pad_char.?) return error.InvalidCharacter;
                    leftover_idx = src_idx;
                    break;
                }

                acc = ((acc << 6) | d) & 0xfff;
                acc_len += 6;
                if (acc_len >= 8) {
                    if (dest_idx == dest.len) return error.NoSpaceLeft;
                    acc_len -= 8;
                    dest[dest_idx] = @truncate(acc >> @as(u5, @intCast(acc_len)));
                    dest_idx += 1;
                }
            }

            if (acc_len > 4 or (acc & ((@as(u32, 1) << @as(u5, @intCast(acc_len))) - 1)) != 0) {
                return error.InvalidPadding;
            }

            const padding_len = acc_len / 2;
            if (leftover_idx == null) {
                if (pad_char != null and padding_len != 0) return error.InvalidPadding;
                return dest_idx;
            }

            const leftover = source[leftover_idx.?..];
            if (comptime pad_char) |p| {
                var padding_chars: usize = 0;

                for (leftover) |c| {
                    if (self.char_is_ignored[c]) continue;
                    if (c != p) {
                        return if (c == invalid_char) error.InvalidCharacter else error.InvalidPadding;
                    }
                    padding_chars += 1;
                }
                if (padding_chars != padding_len) return error.InvalidPadding;
            }

            return dest_idx;
        }
    };
}

/// One Base64 variant: alphabet + padding + a specialized encoder.
///
/// Mirrors std.base64.Codecs.
pub fn Codecs(comptime alphabet: [64]u8, comptime pad: ?u8) type {
    return struct {
        pub const alphabet_chars = alphabet;
        pub const pad_char = pad;

        /// Zero-field struct: `standard.Encoder` is both the type's
        /// namespace and a callable value, matching std's usage
        /// `standard.Encoder.encode(dest, src)`.
        pub const Encoder: EncoderImpl(alphabet, pad) = .{};

        pub const Decoder: DecoderImpl(alphabet, pad) = .{};

        /// Returns a decoder that additionally skips `ignore_chars` (e.g. "\r\n"
        /// for PEM-wrapped input) instead of treating them as InvalidCharacter.
        ///
        /// `ignore_chars` is a runtime value, so `char_is_ignored` is built at
        /// call time, not comptime, same cost shape as std's `init`.
        pub fn decoderWithIgnore(ignore_chars: []const u8) DecoderWithIgnoreImpl(alphabet, pad) {
            return DecoderWithIgnoreImpl(alphabet, pad).init(ignore_chars);
        }
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

fn testRoundTripComptimeSafe(comptime Codec: type, decoded: []const u8, encoded: []const u8) !void {
    // raw encode (stack buffer, no allocator) — safe at comptime
    {
        const out_len = Codec.Encoder.calcSize(decoded.len);
        var buf: [256]u8 = undefined;
        const n = Codec.Encoder.encodeToSlice(buf[0..out_len], decoded);
        try testing.expectEqualStrings(encoded, buf[0..n]);
    }

    // stream encode into a .fixed writer — no allocator, comptime-safe
    {
        var buffer: [256]u8 = undefined;
        var writer: std.Io.Writer = .fixed(&buffer);
        try Codec.Encoder.encodeWriter(&writer, decoded);
        try testing.expectEqualStrings(encoded, writer.buffered());
    }

    // Decoder

    {
        var buffer: [256]u8 = undefined;
        const dest = buffer[0..try Codec.Decoder.calcSizeForSlice(encoded)];
        try Codec.Decoder.decode(dest, encoded);
        try testing.expectEqualStrings(decoded, dest);
    }

    {
        const decoder_ignore_nothing = Codec.decoderWithIgnore("");
        var buffer: [256]u8 = undefined;

        const decoded_buf = buffer[0..decoder_ignore_nothing.calcSizeUpperBound(encoded.len)];
        const written = try decoder_ignore_nothing.decode(decoded_buf, encoded);
        try testing.expect(written <= decoded.len);
        try testing.expectEqualSlices(u8, decoded, decoded_buf[0..written]);
    }
}

fn testAllApis(comptime Codec: type, decoded: []const u8, encoded: []const u8) !void {
    // Encoder

    {
        const out_len = Codec.Encoder.calcSize(decoded.len);
        try std.testing.expectEqual(encoded.len, out_len);

        var buf: [256]u8 = undefined;
        const n = Codec.Encoder.encodeToSlice(buf[0..out_len], decoded);
        try testing.expectEqualStrings(encoded, buf[0..n]);
    }

    {
        const buf = try Codec.Encoder.allocEncode(testing.allocator, decoded);
        defer testing.allocator.free(buf);
        try testing.expectEqualStrings(encoded, buf);
    }

    {
        var aw: std.Io.Writer.Allocating = .init(testing.allocator);
        defer aw.deinit();
        try Codec.Encoder.encodeWriter(&aw.writer, decoded);
        try testing.expectEqualStrings(encoded, aw.written());
    }

    // Decoder
    {
        var buffer: [256]u8 = undefined;
        const dest = buffer[0..try Codec.Decoder.calcSizeForSlice(encoded)];
        try Codec.Decoder.decode(dest, encoded);
        try testing.expectEqualStrings(decoded, dest);
    }

    {
        const decoder_ignore_nothing = Codec.decoderWithIgnore("");
        var buffer: [256]u8 = undefined;

        const decoded_buf = buffer[0..decoder_ignore_nothing.calcSizeUpperBound(encoded.len)];
        const written = try decoder_ignore_nothing.decode(decoded_buf, encoded);
        try testing.expect(written <= decoded.len);
        try testing.expectEqualSlices(u8, decoded, decoded_buf[0..written]);
    }
}

fn testDecodeIgnoreSpace(comptime Codec: type, expected_decoded: []const u8, encoded: []const u8) !void {
    const decoder_ignore_space = Codec.decoderWithIgnore(" ");
    var buffer: [256]u8 = undefined;

    const decoded = buffer[0..decoder_ignore_space.calcSizeUpperBound(encoded.len)];
    const written = try decoder_ignore_space.decode(decoded, encoded);
    try testing.expectEqualSlices(u8, expected_decoded, decoded[0..written]);
}

fn testError(comptime Codec: type, encoded: []const u8, expected: anyerror) !void {
    var buffer: [0x100]u8 = undefined;
    if (Codec.Decoder.calcSizeForSlice(encoded)) |decoded_size| {
        const decoded = buffer[0..decoded_size];
        if (Codec.Decoder.decode(decoded, encoded)) |_| {
            return error.ExpectedError;
        } else |err| if (err != expected) return err;
    } else |err| if (err != expected) return err;

    const decoder_ignore_space = Codec.decoderWithIgnore(" ");
    if (decoder_ignore_space.decode(buffer[0..], encoded)) |_| {
        return error.ExpectedError;
    } else |err| if (err != expected) return err;
}

fn testNoSpaceLeftError(comptime Codec: type, encoded: []const u8) !void {
    const decoder_ignore_space = Codec.decoderWithIgnore(" ");

    var buffer: [256]u8 = undefined;
    const decoded = buffer[0 .. (try Codec.Decoder.calcSizeForSlice(encoded)) - 1];

    if (decoder_ignore_space.decode(decoded, encoded)) |_| {
        return error.ExpectedError;
    } else |err| if (err != error.NoSpaceLeft) return err;
}

fn testFourBytesDestNoSpaceLeftError(comptime Codec: type, encoded: []const u8) !void {
    const decoder_ignore_space = Codec.decoderWithIgnore(" ");

    var buffer: [256]u8 = undefined;
    const decoded = buffer[0..4];

    if (decoder_ignore_space.decode(decoded, encoded)) |_| {
        return error.ExpectedError;
    } else |err| if (err != error.NoSpaceLeft) return err;
}

test "Base64: full std-parity matrix, standard" {
    const Codec = standard;

    try testAllApis(Codec, "", "");
    try testAllApis(Codec, "f", "Zg==");
    try testAllApis(Codec, "fo", "Zm8=");
    try testAllApis(Codec, "foo", "Zm9v");
    try testAllApis(Codec, "foob", "Zm9vYg==");
    try testAllApis(Codec, "fooba", "Zm9vYmE=");
    try testAllApis(Codec, "foobar", "Zm9vYmFy");
    try testAllApis(Codec, "foobarfoobarfoo", "Zm9vYmFyZm9vYmFyZm9v");
    try testAllApis(Codec, "foobarfoobarfoob", "Zm9vYmFyZm9vYmFyZm9vYg==");
    try testAllApis(Codec, "foobarfoobarfooba", "Zm9vYmFyZm9vYmFyZm9vYmE=");
    try testAllApis(Codec, "foobarfoobarfoobar", "Zm9vYmFyZm9vYmFyZm9vYmFy");

    try testDecodeIgnoreSpace(Codec, "", " ");
    try testDecodeIgnoreSpace(Codec, "f", "Z g= =");
    try testDecodeIgnoreSpace(Codec, "fo", "    Zm8=");
    try testDecodeIgnoreSpace(Codec, "foo", "Zm9v    ");
    try testDecodeIgnoreSpace(Codec, "foob", "Zm9vYg = = ");
    try testDecodeIgnoreSpace(Codec, "fooba", "Zm9v YmE=");
    try testDecodeIgnoreSpace(Codec, "foobar", " Z m 9 v Y m F y ");

    try testError(Codec, "A", error.InvalidPadding);
    try testError(Codec, "AA", error.InvalidPadding);
    try testError(Codec, "AAA", error.InvalidPadding);
    try testError(Codec, "A..A", error.InvalidCharacter);
    try testError(Codec, "AA=A", error.InvalidPadding);
    try testError(Codec, "AA/=", error.InvalidPadding);
    try testError(Codec, "A/==", error.InvalidPadding);
    try testError(Codec, "A===", error.InvalidPadding);
    try testError(Codec, "====", error.InvalidPadding);
    try testError(Codec, "Zm9vYmFyZm9vYmFyA..A", error.InvalidCharacter);
    try testError(Codec, "Zm9vYmFyZm9vYmFyAA=A", error.InvalidPadding);
    try testError(Codec, "Zm9vYmFyZm9vYmFyAA/=", error.InvalidPadding);
    try testError(Codec, "Zm9vYmFyZm9vYmFyA/==", error.InvalidPadding);
    try testError(Codec, "Zm9vYmFyZm9vYmFyA===", error.InvalidPadding);
    try testError(Codec, "A..AZm9vYmFyZm9vYmFy", error.InvalidCharacter);
    try testError(Codec, "Zm9vYmFyZm9vAA=A", error.InvalidPadding);
    try testError(Codec, "Zm9vYmFyZm9vAA/=", error.InvalidPadding);
    try testError(Codec, "Zm9vYmFyZm9vA/==", error.InvalidPadding);
    try testError(Codec, "Zm9vYmFyZm9vA===", error.InvalidPadding);

    try testNoSpaceLeftError(Codec, "AA==");
    try testNoSpaceLeftError(Codec, "AAA=");
    try testNoSpaceLeftError(Codec, "AAAA");
    try testNoSpaceLeftError(Codec, "AAAAAA==");

    try testFourBytesDestNoSpaceLeftError(Codec, "AAAAAAAAAAAAAAAA");
}

test "Base64: full std-parity matrix, url_safe_no_pad" {
    const Codec = url_safe_no_pad;

    try testAllApis(Codec, "", "");
    try testAllApis(Codec, "f", "Zg");
    try testAllApis(Codec, "fo", "Zm8");
    try testAllApis(Codec, "foo", "Zm9v");
    try testAllApis(Codec, "foob", "Zm9vYg");
    try testAllApis(Codec, "fooba", "Zm9vYmE");
    try testAllApis(Codec, "foobar", "Zm9vYmFy");
    try testAllApis(Codec, "foobarfoobarfoobar", "Zm9vYmFyZm9vYmFyZm9vYmFy");

    try testDecodeIgnoreSpace(Codec, "", " ");
    try testDecodeIgnoreSpace(Codec, "f", "Z g ");
    try testDecodeIgnoreSpace(Codec, "fo", "    Zm8");
    try testDecodeIgnoreSpace(Codec, "foo", "Zm9v    ");
    try testDecodeIgnoreSpace(Codec, "foob", "Zm9vYg   ");
    try testDecodeIgnoreSpace(Codec, "fooba", "Zm9v YmE");
    try testDecodeIgnoreSpace(Codec, "foobar", " Z m 9 v Y m F y ");

    try testError(Codec, "A", error.InvalidPadding);
    try testError(Codec, "AAA=", error.InvalidCharacter);
    try testError(Codec, "A..A", error.InvalidCharacter);
    try testError(Codec, "AA=A", error.InvalidCharacter);
    try testError(Codec, "AA/=", error.InvalidCharacter);
    try testError(Codec, "A/==", error.InvalidCharacter);
    try testError(Codec, "A===", error.InvalidCharacter);
    try testError(Codec, "====", error.InvalidCharacter);
    try testError(Codec, "Zm9vYmFyZm9vYmFyA..A", error.InvalidCharacter);
    try testError(Codec, "A..AZm9vYmFyZm9vYmFy", error.InvalidCharacter);

    try testNoSpaceLeftError(Codec, "AA");
    try testNoSpaceLeftError(Codec, "AAA");
    try testNoSpaceLeftError(Codec, "AAAA");
    try testNoSpaceLeftError(Codec, "AAAAAA");

    try testFourBytesDestNoSpaceLeftError(Codec, "AAAAAAAAAAAAAAAA");
}

test "Base64Encoder: round trip, standard (comptime)" {
    try comptime testRoundTripComptimeSafe(standard, "foobar", "Zm9vYmFy");
}

test "Base64Encoder: fixed vectors, standard" {
    try testAllApis(standard, "", "");
    try testAllApis(standard, "\x00\x10\x83\x10\x51\x87\x20\x92\x8b", "ABCDEFGHIJKL");
}

test "Base64Encoder: fixed vectors, standard_no_pad" {
    try testAllApis(standard_no_pad, "", "");
    try testAllApis(standard_no_pad, "f", "Zg");
    try testAllApis(standard_no_pad, "fo", "Zm8");
    try testAllApis(standard_no_pad, "foo", "Zm9v");
    try testAllApis(standard_no_pad, "foob", "Zm9vYg");
    try testAllApis(standard_no_pad, "fooba", "Zm9vYmE");
    try testAllApis(standard_no_pad, "foobar", "Zm9vYmFy");
}

test "Base64Encoder: fixed vectors, url_safe" {
    try testAllApis(url_safe, "", "");
    try testAllApis(url_safe, "f", "Zg==");
    try testAllApis(url_safe, "fo", "Zm8=");
    try testAllApis(url_safe, "foo", "Zm9v");
    try testAllApis(url_safe, "foob", "Zm9vYg==");
    try testAllApis(url_safe, "fooba", "Zm9vYmE=");
    try testAllApis(url_safe, "foobar", "Zm9vYmFy");
    try testAllApis(url_safe, "\xfb\xff\xbf", "-_-_");
    try testAllApis(url_safe, "\xff\xff\xff", "____");
}

test "Base64Encoder: compare to std.base64, all codecs" {
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
    try testAllApis(standard, "\xff\xff\xff", "////");
    try testAllApis(url_safe, "\xff\xff\xff", "____");

    // Alternating 62/63 within one 4-sextet group; catches an off-by-one
    // between the adjacent top-of-range branches that an all-63 vector
    // can't distinguish.
    try testAllApis(standard, "\xfb\xff\xbf", "+/+/");
    try testAllApis(url_safe, "\xfb\xff\xbf", "-_-_");

    // All sextets = 0 and = 26/52 pin the other alphabet-offset boundaries
    // (upper/lower/digit transitions in the offset ladder).
    try testAllApis(standard, "\x00\x00\x00", "AAAA");

    // the standard alphabet (0-25 = A-Z, 26-51 = a-z, 52-61 = digits, 62 = +, 63 = /):
    // 0x68 0x1a 0x69 = 01101000 00011010 01101001
    try testAllApis(standard, "\x68\x1a\x69", "aBpp");
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
