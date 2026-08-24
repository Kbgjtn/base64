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
//!   * 1 leftover byte  -> 2 chars + "=="
//!   * 2 leftover bytes -> 3 chars + "="
//!
//! 3 bytes were chosen as the block size because it's the smallest number
//! of bytes whose total bit count (n * 8) is evenly divisible by 6:
//! the smallest n solving `8n % 6 == 0` is n = 3, giving 24 bits -> 4
//! six-bit characters.

const std = @import("std");

const six_bit_mask: u8 = 0x3F;
const base64_alphabet_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Compute the encoded length for a given source length, assuming padding is always used.
fn computeSize(source_len: usize) usize {
    return ((source_len + 2) / 3) * 4;
}

/// Return the exact decoded size for a slice.
/// `source.len` must be a multiple of 4; returns `error.InvalidPadding` otherwise.
fn computeSizeForSlice(source: []const u8) !usize {
    if (source.len % 4 != 0) return error.InvalidPadding;
    var padding: usize = 0;
    if (source.len >= 1 and source[source.len - 1] == '=') padding += 1;
    if (source.len >= 2 and source[source.len - 2] == '=') padding += 1;
    return (source.len / 4) * 3 - padding;
}

/// A Base64 encoder that supports both buffer-based and allocating encoding.
pub const Encoder = struct {
    pub const Error = error{NoSpaceLeft};

    /// Encodes `source` into the caller-provided buffer `dest`, returning
    /// the number of bytes written.
    ///
    /// `dest` must be at least `computeSize(source.len)` bytes long, otherwise
    /// `error.NoSpaceLeft` is returned. Use `calcSize` to compute the
    /// required length ahead of time. `dest` may be larger than required;
    /// callers may reuse an oversized scratch buffer across calls.
    pub fn encodeToSlice(dest: []u8, source: []const u8) !usize {
        if (source.len == 0) return 0;

        const dest_size = computeSize(source.len);
        if (dest.len < dest_size) {
            @branchHint(.unlikely);
            return error.NoSpaceLeft;
        }

        var idx: usize = 0;
        var out_idx: usize = 0;

        // Tier 1: 16 source bytes -> 16 dest chars per iteration.
        while (idx + 15 < source.len) : (idx += 12) {
            const bits = std.mem.readInt(u128, source[idx..][0..16], .big);
            inline for (0..16) |i| {
                dest[out_idx + i] = base64_alphabet_chars[@truncate((bits >> (122 - i * 6)) & six_bit_mask)];
            }
            out_idx += 16;
        }

        // Tier 2: 4 source bytes -> 4 dest chars per iteration.
        while (idx + 3 < source.len) : (idx += 3) {
            const bits = std.mem.readInt(u32, source[idx..][0..4], .big);
            dest[out_idx + 0] = base64_alphabet_chars[(bits >> 26) & six_bit_mask];
            dest[out_idx + 1] = base64_alphabet_chars[(bits >> 20) & six_bit_mask];
            dest[out_idx + 2] = base64_alphabet_chars[(bits >> 14) & six_bit_mask];
            dest[out_idx + 3] = base64_alphabet_chars[(bits >> 8) & six_bit_mask];
            out_idx += 4;
        }

        // Scalar tail: 0-2 leftover bytes (a full 3-byte remainder would
        // already have been consumed by the tier-2 loop above).
        if (idx + 2 < source.len) {
            dest[out_idx + 0] = base64_alphabet_chars[source[idx] >> 2];
            dest[out_idx + 1] = base64_alphabet_chars[((source[idx] & 0x3) << 4) | (source[idx + 1] >> 4)];
            dest[out_idx + 2] = base64_alphabet_chars[((source[idx + 1] & 0xf) << 2) | (source[idx + 2] >> 6)];
            dest[out_idx + 3] = base64_alphabet_chars[source[idx + 2] & six_bit_mask];
            out_idx += 4;
        } else if (idx + 1 < source.len) {
            dest[out_idx + 0] = base64_alphabet_chars[source[idx] >> 2];
            dest[out_idx + 1] = base64_alphabet_chars[((source[idx] & 0x3) << 4) | (source[idx + 1] >> 4)];
            dest[out_idx + 2] = base64_alphabet_chars[(source[idx + 1] & 0xf) << 2];
            out_idx += 3;
        } else if (idx < source.len) {
            dest[out_idx + 0] = base64_alphabet_chars[source[idx] >> 2];
            dest[out_idx + 1] = base64_alphabet_chars[(source[idx] & 0x3) << 4];
            out_idx += 2;
        }

        // Batched pad-fill instead of inline per-case '=' writes.
        for (dest[out_idx..dest_size]) |*pad| {
            pad.* = '=';
        }

        return dest_size;
    }

    /// Encodes `source` into a newly allocated buffer using `allocator`.
    ///
    /// The caller owns the returned slice and is responsible for freeing it.
    pub fn encode(gpa: std.mem.Allocator, source: []const u8) ![]u8 {
        const size = computeSize(source.len);
        const out = try gpa.alloc(u8, size);
        errdefer gpa.free(out);
        const n = try encodeToSlice(out, source);
        return out[0..n];
    }
};

/// A Base64 decoder that supports both buffer-based and allocating encoding.
pub const Decoder = struct {
    const invalid_char_tst: u32 = 0xFF000000;
    const invalid_char: u8 = 0xFF;

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

    const encoded_n = try Encoder.encodeToSlice(encoded, input);
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
        const n = try Encoder.encodeToSlice(dest, input);
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

        const n = try Encoder.encodeToSlice(dest, input);
        try std.testing.expectEqual(dest.len, n);

        // Round-trip through the decoder as an independent correctness
        // check on the encoded bytes, rather than re-deriving expected
        // base64 text by hand for every size.
        const decoded = try Decoder.decode(std.testing.allocator, dest, .reject);
        defer std.testing.allocator.free(decoded);
        try std.testing.expectEqualSlices(u8, input, decoded);
    }
}

test "Encoder rejects undersized dest" {
    // Mirrors std.base64's testNoSpaceLeftError: dest one byte short of
    // the required size must fail, not overflow or silently truncate.
    const cases = [_][]const u8{ "f", "fo", "foo", "foob", "fooba", "foobar" };

    for (cases) |input| {
        const required = computeSize(input.len);
        if (required == 0) continue;

        var buf: [128]u8 = undefined;
        const dest = buf[0 .. required - 1];
        try std.testing.expectError(error.NoSpaceLeft, Encoder.encodeToSlice(dest, input));
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
    const exact_n = try Encoder.encodeToSlice(exact[0..required], input);

    var oversized: [128]u8 = undefined;
    @memset(&oversized, 0xAA);
    const oversized_n = try Encoder.encodeToSlice(&oversized, input);

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
        _ = try Encoder.encodeToSlice(encoded, input);

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
