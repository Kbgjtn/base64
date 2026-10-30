//! Fuzz targets for the four codecs, checked against `std.base64`, the oracle the README
//! names. Two targets per codec:
//!
//! - "decode": any bytes through `Decoder` and `decoderWithIgnore`. The size functions and
//!   every result, error or bytes, must match std's. Text that decodes must encode back to
//!   itself, since only the canonical encoding decodes (RFC 4648 §3.5), and `decode` must
//!   not write past `calcSizeForSlice`.
//! - "encode": any bytes through `encodeToSlice`, `encode`, `allocEncode` and
//!   `encodeWriter`. All must equal std's output and `calcSize`, write nothing past it, and
//!   decode back to the input, also with CRLF line breaks every 76 characters (RFC 2045
//!   §6.8) through `decoderWithIgnore`. A tiled copy of the input, 4 to 40 KiB, checks
//!   `encodeWriter`'s chunked path against `encodeToSlice`.
//!
//! - `zig build test` runs the RFC 4648 §10 vectors, the error vectors from base64.zig and
//!   a fixed set of seeded mutations of them.
//! - `zig build test --fuzz -Dllvm -Dtest-filter="fuzz: standard decode"` hands one target
//!   to the fuzzer. Names: "fuzz: <codec> decode" and "fuzz: <codec> encode".
const std = @import("std");
const base64 = @import("../base64.zig");
const Smith = std.testing.Smith;

const testing = std.testing;
const gpa = std.testing.allocator;

/// Largest fuzz input: past the 32-byte avx2 block and the 16-byte scalar loop many times
/// over, and still fast. The chunked `encodeWriter` path gets a tiled copy instead.
const max_input = 4 << 10;

/// Largest tiled input: `encodeWriter` past `small_in_len` and over several
/// `stream_in_len` chunks, with a remainder.
const max_tiled = base64.small_in_len + 3 * base64.stream_in_len;

/// Bytes after each output that must stay untouched.
const guard_len = 64;
const guard: u8 = 0xa5;

/// What `decoderWithIgnore` skips here: spaces and the line breaks of MIME and PEM text
/// (RFC 4648 §3.3).
const ignore_chars = " \t\r\n";

/// RFC 4648 §10 test vectors and the longer ones from base64.zig, decoded.
const plain_vectors = [_][]const u8{
    "",                 "f",                 "fo",                 "foo",
    "foob",             "fooba",             "foobar",             "foobarfoobarfoo",
    "foobarfoobarfoob", "foobarfoobarfooba", "foobarfoobarfoobar",
};

/// Bad and spaced text from base64.zig's std-parity matrix.
const text_vectors = [_][]const u8{
    "A",                    "AA",                   "AAA",                  "A..A",
    "AA=A",                 "AA/=",                 "A/==",                 "A===",
    "====",                 "AAA=",                 "Zm9vYmFyZm9vYmFyA..A", "Zm9vYmFyZm9vYmFyAA=A",
    "Zm9vYmFyZm9vYmFyAA/=", "Zm9vYmFyZm9vYmFyA/==", "Zm9vYmFyZm9vYmFyA===", "A..AZm9vYmFyZm9vYmFy",
    "Zm9vYmFyZm9vAA=A",     "Zm9vYmFyZm9vAA/=",     "Zm9vYmFyZm9vA/==",     "Zm9vYmFyZm9vA===",
    "AAAAAAAAAAAAAAAA",     " ",                    "Z g= =",               "    Zm8=",
    "Zm9v    ",             "Zm9vYg = = ",          "Zm9v YmE=",            " Z m 9 v Y m F y ",
    "Z g ",                 "    Zm8",              "Zm9vYg   ",            "Zm9v YmE",
};

/// One input's bytes, and the scratch every check uses. The fuzzer runs one input at a
/// time, so one set will do; the outputs are too big for the stack of a fuzz worker.
var input_buf: [max_input]u8 = undefined;
var tiled_buf: [max_tiled]u8 = undefined;
var want_buf: [base64.standard.Encoder.calcSize(max_tiled) + guard_len]u8 = undefined;
var got_buf: [base64.standard.Encoder.calcSize(max_tiled) + guard_len]u8 = undefined;
var wrap_buf: [base64.standard.Encoder.calcSize(max_input) * 2]u8 = undefined;

/// Every codec, paired with its std oracle.
const codecs = .{
    .{ base64.standard, std.base64.standard },
    .{ base64.standard_no_pad, std.base64.standard_no_pad },
    .{ base64.url_safe, std.base64.url_safe },
    .{ base64.url_safe_no_pad, std.base64.url_safe_no_pad },
};

fn Targets(comptime Codec: type, comptime oracle: std.base64.Codecs) type {
    return struct {
        fn oracleEncode(dest: []u8, bytes: []const u8) []const u8 {
            return oracle.Encoder.encode(dest, bytes);
        }

        fn decodeAnything(_: void, smith: *Smith) anyerror!void {
            const len = smith.slice(&input_buf);
            _ = try checkDecode(input_buf[0..len]);
        }

        fn encodeAnything(_: void, smith: *Smith) anyerror!void {
            const len = smith.slice(&input_buf);
            try checkEncode(input_buf[0..len]);
            try checkTiled(input_buf[0..len]);
        }

        /// Decodes `text` both ways and compares. Returns whether it decoded.
        fn checkDecode(text: []const u8) !bool {
            std.debug.assert(text.len <= max_input);
            try expectSameSize(
                oracle.Decoder.calcSizeUpperBound(text.len),
                Codec.Decoder.calcSizeUpperBound(text.len),
            );
            try expectSameSize(oracle.Decoder.calcSizeForSlice(text), Codec.Decoder.calcSizeForSlice(text));
            try checkIgnore(text);

            const size = Codec.Decoder.calcSizeForSlice(text) catch return false;
            try testing.expect(size <= try Codec.Decoder.calcSizeUpperBound(text.len));

            @memset(got_buf[0 .. size + guard_len], guard);
            const want = oracle.Decoder.decode(want_buf[0..size], text);
            const got = Codec.Decoder.decode(got_buf[0..size], text);
            try expectGuard(got_buf[size..][0..guard_len]);
            want catch |err| {
                try testing.expectError(err, got);
                return false;
            };
            try got;
            try testing.expectEqualSlices(u8, want_buf[0..size], got_buf[0..size]);

            // Canonical: only one text decodes to these bytes, so encoding gives it back.
            var again: [max_input]u8 = undefined;
            try testing.expectEqualStrings(text, Codec.Encoder.encode(&again, got_buf[0..size]));
            return true;
        }

        /// `decoderWithIgnore` against std's: the bound, and the result with room for the
        /// bound and with one byte less than std wrote.
        fn checkIgnore(text: []const u8) !void {
            // Not `oracle.decoderWithIgnore`: in Zig 0.16.0 std's codecs share one per
            // alphabet, so `standard_no_pad`'s expects padding and `url_safe`'s refuses it.
            // Fixed upstream by codeberg.org/ziglang/zig/pulls/37093.
            const theirs: std.base64.Base64DecoderWithIgnore = .init(oracle.alphabet_chars, oracle.pad_char, ignore_chars);
            const mine = Codec.decoderWithIgnore(ignore_chars);
            const bound = mine.calcSizeUpperBound(text.len);
            try testing.expectEqual(theirs.calcSizeUpperBound(text.len), bound);

            const want_len = try expectSameDecode(theirs, mine, text, bound);
            if (want_len) |n| if (n > 0) {
                _ = try expectSameDecode(theirs, mine, text, n - 1);
            };
        }

        /// Runs both ignore decoders on `text` into `dest_len` bytes and compares. Returns
        /// how many bytes std wrote, or null if it failed.
        fn expectSameDecode(theirs: anytype, mine: anytype, text: []const u8, dest_len: usize) !?usize {
            @memset(got_buf[0 .. dest_len + guard_len], guard);
            const want = theirs.decode(want_buf[0..dest_len], text);
            const got = mine.decode(got_buf[0..dest_len], text);
            try expectGuard(got_buf[dest_len..][0..guard_len]);
            const n = want catch |err| {
                try testing.expectError(err, got);
                return null;
            };
            try testing.expectEqual(n, try got);
            try testing.expectEqualSlices(u8, want_buf[0..n], got_buf[0..n]);
            return n;
        }

        fn checkEncode(bytes: []const u8) !void {
            const size = Codec.Encoder.calcSize(bytes.len);
            try testing.expectEqual(oracle.Encoder.calcSize(bytes.len), size);
            const want = oracle.Encoder.encode(&want_buf, bytes);
            try testing.expectEqual(size, want.len);

            // A dest longer than needed: nothing past `size` may change.
            @memset(got_buf[0 .. size + guard_len], guard);
            try testing.expectEqual(size, Codec.Encoder.encodeToSlice(got_buf[0 .. size + guard_len], bytes));
            try testing.expectEqualStrings(want, got_buf[0..size]);
            try expectGuard(got_buf[size..][0..guard_len]);

            try testing.expectEqualStrings(want, Codec.Encoder.encode(got_buf[0..size], bytes));

            const allocated = try Codec.Encoder.allocEncode(gpa, bytes);
            defer gpa.free(allocated);
            try testing.expectEqualStrings(want, allocated);

            // Exactly `size` bytes of room: one more write would fail.
            var fixed: std.Io.Writer = .fixed(got_buf[0..size]);
            try Codec.Encoder.encodeWriter(&fixed, bytes);
            try testing.expectEqualStrings(want, fixed.buffered());

            // And back.
            try testing.expectEqual(bytes.len, try Codec.Decoder.calcSizeForSlice(want));
            try Codec.Decoder.decode(got_buf[0..bytes.len], want);
            try testing.expectEqualSlices(u8, bytes, got_buf[0..bytes.len]);

            // Wrapped at 76 characters with CRLF, as MIME does (RFC 2045 §6.8).
            var wrapped: std.Io.Writer = .fixed(&wrap_buf);
            var rest = want;
            while (rest.len > 0) {
                const line = rest[0..@min(76, rest.len)];
                try wrapped.writeAll(line);
                try wrapped.writeAll("\r\n");
                rest = rest[line.len..];
            }
            const ignoring = Codec.decoderWithIgnore(ignore_chars);
            const text = wrapped.buffered();
            const bound = ignoring.calcSizeUpperBound(text.len);
            const n = try ignoring.decode(got_buf[0..bound], text);
            try testing.expectEqualSlices(u8, bytes, got_buf[0..n]);
        }

        /// `encodeWriter` on `bytes` repeated out to 4 to 40 KiB, past `small_in_len` and
        /// across `stream_in_len` chunks, against `encodeToSlice`.
        fn checkTiled(bytes: []const u8) !void {
            if (bytes.len == 0) return;
            const extra = (@as(usize, bytes[0]) * 97 + bytes.len) % (max_tiled - base64.small_in_len);
            const tiled = tiled_buf[0 .. base64.small_in_len + 1 + extra];
            for (tiled, 0..) |*b, i| b.* = bytes[i % bytes.len];

            const size = Codec.Encoder.calcSize(tiled.len);
            try testing.expectEqual(size, Codec.Encoder.encodeToSlice(want_buf[0..size], tiled));
            var fixed: std.Io.Writer = .fixed(got_buf[0..size]);
            try Codec.Encoder.encodeWriter(&fixed, tiled);
            try testing.expectEqualStrings(want_buf[0..size], fixed.buffered());
        }
    };
}

/// Both size results must be the same number or the same error.
fn expectSameSize(want: std.base64.Error!usize, got: base64.Error!usize) !void {
    const n = want catch |err| return testing.expectError(err, got);
    try testing.expectEqual(n, try got);
}

fn expectGuard(bytes: []const u8) !void {
    for (bytes) |b| try testing.expectEqual(guard, b);
}

/// Adds `bytes` to `corpus` as a `Smith` slice input: a little-endian u32 length, then the
/// bytes.
fn addInput(corpus: *std.ArrayList([]const u8), bytes: []const u8) !void {
    const input = try gpa.alloc(u8, 4 + bytes.len);
    std.mem.writeInt(u32, input[0..4], @intCast(bytes.len), .little);
    @memcpy(input[4..], bytes);
    corpus.append(gpa, input) catch |err| {
        gpa.free(input);
        return err;
    };
}

fn freeCorpus(corpus: *std.ArrayList([]const u8)) void {
    for (corpus.items) |input| gpa.free(input);
    corpus.deinit(gpa);
}

/// Lengths of the random seeds: around the avx2 block (32 characters, 24 bytes), the
/// simd128 block (16 and 12) and the scalar 16-character loop, then longer runs. The
/// vectors alone stop at 24 characters, and the fuzzer seldom grows an input past that.
const seed_lens = [_]usize{ 11, 12, 13, 23, 24, 25, 31, 32, 33, 47, 48, 49, 96, 200, 1000, 3000 };

/// Fills `buf` with the same pseudo-random bytes every run.
fn seedBytes(buf: []u8) []u8 {
    var prng: std.Random.DefaultPrng = .init(buf.len);
    prng.random().bytes(buf);
    return buf;
}

/// Decode seeds for one codec: the plain vectors and `seed_lens` random bytes encoded with
/// it, then `text_vectors`.
fn decodeSeeds(comptime oracle: std.base64.Codecs, corpus: *std.ArrayList([]const u8)) !void {
    var buf: [max_input]u8 = undefined;
    var raw: [seed_lens[seed_lens.len - 1]]u8 = undefined;
    for (plain_vectors) |plain| try addInput(corpus, oracle.Encoder.encode(&buf, plain));
    for (seed_lens) |len| try addInput(corpus, oracle.Encoder.encode(&buf, seedBytes(raw[0..len])));
    for (text_vectors) |text| try addInput(corpus, text);
}

/// Encode seeds: the plain vectors and `seed_lens` random bytes.
fn encodeSeeds(corpus: *std.ArrayList([]const u8)) !void {
    var raw: [seed_lens[seed_lens.len - 1]]u8 = undefined;
    for (plain_vectors) |plain| try addInput(corpus, plain);
    for (seed_lens) |len| try addInput(corpus, seedBytes(raw[0..len]));
}

fn fuzzDecode(comptime pair: anytype) !void {
    var corpus: std.ArrayList([]const u8) = .empty;
    defer freeCorpus(&corpus);
    try decodeSeeds(pair[1], &corpus);
    try testing.fuzz({}, Targets(pair[0], pair[1]).decodeAnything, .{ .corpus = corpus.items });
}

fn fuzzEncode(comptime pair: anytype) !void {
    var corpus: std.ArrayList([]const u8) = .empty;
    defer freeCorpus(&corpus);
    try encodeSeeds(&corpus);
    try testing.fuzz({}, Targets(pair[0], pair[1]).encodeAnything, .{ .corpus = corpus.items });
}

test "fuzz: standard decode" {
    try fuzzDecode(codecs[0]);
}

test "fuzz: standard encode" {
    try fuzzEncode(codecs[0]);
}

test "fuzz: standard_no_pad decode" {
    try fuzzDecode(codecs[1]);
}

test "fuzz: standard_no_pad encode" {
    try fuzzEncode(codecs[1]);
}

test "fuzz: url_safe decode" {
    try fuzzDecode(codecs[2]);
}

test "fuzz: url_safe encode" {
    try fuzzEncode(codecs[2]);
}

test "fuzz: url_safe_no_pad decode" {
    try fuzzDecode(codecs[3]);
}

test "fuzz: url_safe_no_pad encode" {
    try fuzzEncode(codecs[3]);
}

test "fuzz seeded: decode 240 mutations of each seed, all codecs" {
    // The decode targets on fixed pseudo-random damage, so plain `zig build test` covers
    // them. Seeds: the vectors, and std's encoding of random bytes of every length to 200,
    // which reaches the SIMD blocks and the 16-byte loop.
    var prng: std.Random.DefaultPrng = .init(0xB64_2026);
    const random = prng.random();
    var seed_buf: [max_input]u8 = undefined;
    var mutated_buf: [max_input]u8 = undefined;
    var raw: [200]u8 = undefined;
    inline for (codecs) |pair| {
        const T = Targets(pair[0], pair[1]);
        var corpus: std.ArrayList([]const u8) = .empty;
        defer freeCorpus(&corpus);
        try decodeSeeds(pair[1], &corpus);
        for (0..raw.len + 1) |len| {
            random.bytes(raw[0..len]);
            try addInput(&corpus, T.oracleEncode(&seed_buf, raw[0..len]));
        }

        var decoded: usize = 0;
        for (corpus.items) |input| {
            const seed = input[4..];
            if (try T.checkDecode(seed)) decoded += 1;
            for (0..240) |_| {
                if (try T.checkDecode(mutate(random, seed, &mutated_buf))) decoded += 1;
            }
        }
        // Guards the mutator: if nearly every input failed to decode, the success path
        // (bytes, canonical re-encode) would go untested. 7,528 to 8,841 of the 62,660
        // inputs decode from this seed, depending on the codec.
        try testing.expect(decoded >= 6000);
    }
}

test "fuzz seeded: encode random bytes of every length to 600, all codecs" {
    // The encode targets on fixed pseudo-random input, so plain `zig build test` covers
    // them: every tail length against every SIMD block size, and a tiled copy of each.
    var prng: std.Random.DefaultPrng = .init(0xB64_2027);
    const random = prng.random();
    var raw: [600]u8 = undefined;
    inline for (codecs) |pair| {
        const T = Targets(pair[0], pair[1]);
        for (plain_vectors) |plain| try T.checkEncode(plain);
        for (0..raw.len + 1) |len| {
            random.bytes(raw[0..len]);
            try T.checkEncode(raw[0..len]);
            try T.checkTiled(raw[0..len]);
        }
    }
}

/// Characters a mutation writes: both alphabets' edge characters, padding, the ignored
/// characters, and bytes no codec accepts.
const mutation_chars = "AZaz09+/-_==  \r\n\t.\x00\x80\xff";

/// A damaged copy of `text` in `buf`: a few characters replaced, then maybe cut short, or
/// a run duplicated in place, which shifts every block after it.
fn mutate(random: std.Random, text: []const u8, buf: *[max_input]u8) []const u8 {
    var len = text.len;
    @memcpy(buf[0..len], text);
    if (len == 0) return buf[0..0];
    for (0..random.intRangeAtMost(u8, 1, 3)) |_| {
        buf[random.uintLessThan(usize, len)] = if (random.boolean())
            mutation_chars[random.uintLessThan(usize, mutation_chars.len)]
        else
            random.int(u8);
    }
    switch (random.uintLessThan(u8, 4)) {
        0 => len = random.uintLessThan(usize, len),
        1 => {
            const at = random.uintLessThan(usize, len);
            const run = @min(random.intRangeAtMost(usize, 1, 8), len - at, buf.len - len);
            std.mem.copyBackwards(u8, buf[at + run .. len + run], buf[at..len]);
            len += run;
        },
        else => {},
    }
    return buf[0..len];
}
