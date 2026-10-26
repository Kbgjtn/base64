//! Usage: bench [--corpus DIR] [--csv] [--runs N] [--target-ms N]
//!              [--filter SUBSTR] [--op NAME]

const std = @import("std");
const builtin = @import("builtin");

const base64 = @import("base64.zig");

const Io = std.Io;

const Variant = enum {
    raw,
    std,
    std_nopad,
    url,
    url_nopad,
    std_crlf76,
    err_late,
    err_pad,
};

const Case = struct {
    name: []const u8,
    files: std.EnumArray(Variant, ?[]const u8),

    fn get(self: *const Case, v: Variant) ?[]const u8 {
        return self.files.get(v);
    }
};

/// One codec pair: ours, std's, and the corpus variant that holds its encoding.
fn CodecPair(comptime name: []const u8, comptime Mine: type, comptime theirs: std.base64.Codecs, comptime variant: Variant) type {
    return struct {
        const codec_name = name;
        const encoded_variant = variant;

        fn mineEncode(out: []u8, in: []const u8) void {
            std.mem.doNotOptimizeAway(Mine.Encoder.encode(out, in).ptr);
        }

        fn stdEncode(out: []u8, in: []const u8) void {
            std.mem.doNotOptimizeAway(theirs.Encoder.encode(out, in).ptr);
        }

        fn mineDecode(out: []u8, in: []const u8) void {
            Mine.Decoder.decode(out, in) catch {};
            std.mem.doNotOptimizeAway(out.ptr);
        }

        fn stdDecode(out: []u8, in: []const u8) void {
            theirs.Decoder.decode(out, in) catch {};
            std.mem.doNotOptimizeAway(out.ptr);
        }
    };
}

const codec_pairs = .{
    CodecPair("standard", base64.standard, std.base64.standard, .std),
    CodecPair("standard_no_pad", base64.standard_no_pad, std.base64.standard_no_pad, .std_nopad),
    CodecPair("url_safe", base64.url_safe, std.base64.url_safe, .url),
    CodecPair("url_safe_no_pad", base64.url_safe_no_pad, std.base64.url_safe_no_pad, .url_nopad),
};

/// `decodeWithIgnore` values are runtime-built (both here and in std),
/// so the kernels read them from globals set up once in `main`.
const crlf = "\r\n";
var mine_ignore: @TypeOf(base64.standard.decoderWithIgnore(crlf)) = undefined;
var std_ignore: std.base64.Base64DecoderWithIgnore = undefined;

fn mineDecodeIgnore(out: []u8, in: []const u8) void {
    std.mem.doNotOptimizeAway(mine_ignore.decode(out, in) catch 0);
    std.mem.doNotOptimizeAway(out.ptr);
}

fn stdDecodeIgnore(out: []u8, in: []const u8) void {
    std.mem.doNotOptimizeAway(std_ignore.decode(out, in) catch 0);
    std.mem.doNotOptimizeAway(out.ptr);
}

const Kernel = fn (out: []u8, in: []const u8) void;

const Options = struct {
    corpus: []const u8 = "bench/corpus",
    csv: bool = false,
    runs: usize = 7,
    target_ns: u64 = 10 * std.time.ns_per_ms,
    filter: ?[]const u8 = null,
    op: ?[]const u8 = null,
};

const max_runs = 64;

const Stats = struct {
    median_ns: f64,
    min_ns: f64,
};

const Result = struct {
    iterations: usize,
    mine: Stats,
    std: Stats,
};

const Env = struct {
    io: Io,
    gpa: std.mem.Allocator,
    out: *Io.Writer,
    opts: Options,
    host_cpu: []const u8,
    mine_buf: []u8,
    std_buf: []u8,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const arena = init.arena.allocator();

    const opts = try parseArgs(init.minimal.args, arena);

    var stdout_buffer: [4 << 10]u8 = undefined;
    var stdout_writer = Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    mine_ignore = base64.standard.decoderWithIgnore(crlf);
    std_ignore = std.base64.standard.decoderWithIgnore(crlf);

    const cases = loadCorpus(io, arena, opts.corpus) catch |err| {
        std.debug.print("error: can't laod corpus from '{s}' ({t}); run `python3 scripts/gen-corpus.py` first\n", .{
            opts.corpus, err,
        });
        return err;
    };

    var max_len: usize = 0;
    for (cases) |c| {
        for (c.files.values) |f|
            if (f) |data| {
                max_len = @max(max_len, data.len);
            };
    }

    // Room for the largest encoded form of the largest file.
    const buf_len = std.base64.standard.Encoder.calcSize(max_len) + 64;

    var env: Env = .{
        .io = io,
        .gpa = arena,
        .out = stdout,
        .opts = opts,
        .host_cpu = hostCpu(io, arena),
        .mine_buf = try gpa.alloc(u8, buf_len),
        .std_buf = try gpa.alloc(u8, buf_len),
    };

    defer gpa.free(env.mine_buf);
    defer gpa.free(env.std_buf);

    try printHeader(&env, cases.len);
    for (cases) |*c| {
        if (opts.filter) |f| if (std.mem.find(u8, c.name, f) == null) continue;

        try runCase(&env, c);
        try stdout.flush();
    }

    try stdout.flush();
}

fn parseArgs(args: std.process.Args, arena: std.mem.Allocator) !Options {
    var opts: Options = .{};
    var it = try args.iterateAllocator(arena);
    _ = it.next();

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--csv")) {
            opts.csv = true;
        } else if (std.mem.eql(u8, arg, "--corpus")) {
            opts.corpus = it.next() orelse return usage();
        } else if (std.mem.eql(u8, arg, "--runs")) {
            opts.runs = try std.fmt.parseInt(usize, it.next() orelse return usage(), 10);
            if (opts.runs == 0 or opts.runs > max_runs) return usage();
        } else if (std.mem.eql(u8, arg, "--target-ms")) {
            const ms = try std.fmt.parseInt(u64, it.next() orelse return usage(), 10);
            opts.target_ns = ms * std.time.ns_per_ms;
        } else if (std.mem.eql(u8, arg, "--filter")) {
            opts.filter = it.next() orelse return usage();
        } else if (std.mem.eql(u8, arg, "--op")) {
            opts.op = it.next() orelse return usage();
        } else {
            return usage();
        }
    }

    return opts;
}

fn usage() error{InvalidArgs} {
    std.debug.print(
        \\usage: bench [--corpus DIR] [--csv] [--runs N (1-64)] [--target-ms N]
        \\             [--filter CASE_SUBSTR] [--op encode|decode|decode_ignore|decode_err_late|decode_err_pad]
        \\
    , .{});
    return error.InvalidArgs;
}

/// Reads `manifest.tsv` and every file it lists. Rows for the same case are
/// grouped in file order.
fn loadCorpus(io: Io, arena: std.mem.Allocator, dir: []const u8) ![]Case {
    const manifest_path = try std.fs.path.join(arena, &.{ dir, "manifest.tsv" });
    const manifest = try Io.Dir.cwd().readFileAlloc(io, manifest_path, arena, .limited(16 << 20));

    var cases: std.ArrayList(Case) = .empty;
    var lines = std.mem.tokenizeScalar(u8, manifest, '\n');

    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;

        var fields = std.mem.splitScalar(u8, line, '\t');
        const case_name = fields.next() orelse return error.BadManifest;
        const variant_name = fields.next() orelse return error.BadManifest;
        const len_str = fields.next() orelse return error.BadManifest;

        _ = fields.next() orelse return error.BadManifest; // sha256, checked by gen-corpus.py --verify

        const rel = fields.next() orelse return error.BadManifest;
        const variant = std.meta.stringToEnum(Variant, variant_name) orelse return error.BadManifest;
        const len = try std.fmt.parseInt(usize, len_str, 10);
        const path = try std.fs.path.join(arena, &.{ dir, rel });

        const data = try Io.Dir.cwd().readFileAllocOptions(io, path, arena, .unlimited, .@"64", null);
        if (data.len != len) return error.CorpusSizeMismatch;
        if (cases.items.len == 0 or !std.mem.eql(u8, cases.items[cases.items.len - 1].name, case_name)) {
            try cases.append(arena, .{ .name = case_name, .files = .initFill(null) });
        }
        cases.items[cases.items.len - 1].files.set(variant, data);
    }
    return cases.items;
}

/// Host CPU model from /proc/cpuinfo (Linux); the *target* CPU is
/// `builtin.cpu.model.name`, which is what `-Dcpu` picked.
fn hostCpu(io: Io, arena: std.mem.Allocator) []const u8 {
    if (builtin.os.tag != .linux) return "unknown";
    // /proc files report size 0, so read them as a stream, not possitionally.
    var file = Io.Dir.cwd().openFile(io, "/proc/cpuinfo", .{}) catch return "unknown";
    defer file.close(io);

    var buf: [4 << 10]u8 = undefined;
    var reader = file.readerStreaming(io, &buf);
    const info = reader.interface.allocRemaining(arena, .limited(1 << 20)) catch return "unknown";

    var lines = std.mem.tokenizeScalar(u8, info, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "model name")) {
            const colon = std.mem.findScalar(u8, line, ':') orelse continue;
            const name = std.mem.trim(u8, line[colon + 1 ..], " \t");
            const copy = arena.dupe(u8, name) catch return "unknown";
            std.mem.replaceScalar(u8, copy, ',', ' ');
            return copy;
        }
    }

    return "unknown";
}

fn printHeader(env: *Env, n_cases: usize) !void {
    const out = env.out;
    if (env.opts.csv) {
        try out.print(
            "tier,target_cpu,host_cpu,mode,case,payload_bytes,input_bytes,op,codec," ++
                "iterations,runs,mine_ns_median,mine_ns_min,std_ns_median,std_ns_min," ++
                "mine_gib_s,std_gib_s,speedup\n",
            .{},
        );
        return;
    }
    try out.print(
        \\Base64 benchmark: this package vs std.base64
        \\  tier:       {t}
        \\  target cpu: {s} (-Dcpu)
        \\  host cpu:   {s}
        \\  mode:       {t}
        \\  zig:        {s}
        \\  corpus:     {s} ({d} cases)
        \\  timing:     median of {d} runs, ~{d} ms each, alternating order
        \\ GiB/s is payload (decoded) bytes per second; speedup = std ns / our ns
        \\
        \\{s:<20} {s:<16} {s:<16} {s:>10} {s:>12} {s:>12} {s:>9} {s:>9} {s:>8}
        \\
    ,
        .{
            base64.tier,
            builtin.cpu.model.name,
            env.host_cpu,
            builtin.mode,
            builtin.zig_version_string,
            env.opts.corpus,
            n_cases,
            env.opts.runs,
            env.opts.target_ns / std.time.ns_per_ms,
            "case",
            "op",
            "codec",
            "bytes",
            "ours ns/op",
            "std ns/op",
            "ours GiB/s",
            "std GiB/s",
            "speedup",
        },
    );
}

fn opEnabled(env: *const Env, op: []const u8) bool {
    const want = env.opts.op orelse return true;
    return std.mem.eql(u8, want, op);
}

fn runCase(env: *Env, c: *const Case) !void {
    const raw = c.get(.raw) orelse return error.CaseWithoutPayload;

    inline for (codec_pairs) |P| {
        const encoded = c.get(P.encoded_variant) orelse return error.UnhandledVariant;
        if (opEnabled(env, "encode")) {
            // The fixture length is the exact calcSize for this codec.
            const exact_len = encoded.len;
            try verifyEncode(
                P,
                c.name,
                raw,
                encoded,
                env.mine_buf[0..exact_len],
                env.std_buf[0..exact_len],
            );

            const r = measure(
                P.mineEncode,
                P.stdEncode,
                env,
                env.mine_buf[0..exact_len],
                env.std_buf[0..exact_len],
                raw,
            );

            try report(env, c.name, raw.len, raw.len, "encode", P.codec_name, r);
        }

        if (opEnabled(env, "decode")) {
            try verifyDecode(
                P,
                c.name,
                raw,
                encoded,
                env.mine_buf[0..raw.len],
                env.std_buf[0..raw.len],
            );

            const r = measure(
                P.mineDecode,
                P.stdDecode,
                env,
                env.mine_buf[0..raw.len],
                env.std_buf[0..raw.len],
                encoded,
            );

            try report(env, c.name, raw.len, encoded.len, "decode", P.codec_name, r);
        }

        // Ignore and error paths only on `standard`: they don't depend on the alphabet.
        if (comptime std.mem.eql(u8, P.codec_name, "standard")) {
            if (opEnabled(env, "decode_ignore")) if (c.get(.std_crlf76)) |wrapped| {
                const cap = std_ignore.calcSizeUpperBound(wrapped.len);
                try verifyDecodeIgnore(
                    c.name,
                    raw,
                    wrapped,
                    env.mine_buf[0..cap],
                    env.std_buf[0..cap],
                );

                const r = measure(mineDecodeIgnore, stdDecodeIgnore, env, env.mine_buf[0..cap], env.std_buf[0..cap], wrapped);
                try report(env, c.name, raw.len, wrapped.len, "decode_ignore", P.codec_name, r);
            };

            inline for (.{ .{ Variant.err_late, "decode_err_late" }, .{ Variant.err_pad, "decode_err_pad" } }) |e| {
                if (opEnabled(env, e[1])) if (c.get(e[0])) |bad| {
                    if (try verifyDecodeError(P, c.name, bad, env.mine_buf, env.std_buf)) |out_len| {
                        const r = measure(P.mineDecode, P.stdDecode, env, env.mine_buf[0..out_len], env.std_buf[0..out_len], bad);
                        try report(env, c.name, raw.len, bad.len, e[1], P.codec_name, r);
                    }
                };
            }
        }
    }
}

fn mismatch(
    case: []const u8,
    op: []const u8,
    codec: []const u8,
    what: []const u8,
) error{VerificationFailed} {
    std.debug.print(
        "verification failed: {s} {s} {s}: {s}\n",
        .{ case, op, codec, what },
    );
    return error.VerificationFailed;
}

fn verifyEncode(
    comptime P: type,
    case: []const u8,
    raw: []const u8,
    fixture: []const u8,
    mine: []u8,
    theirs: []u8,
) !void {
    P.mineEncode(mine, raw);
    P.stdEncode(theirs, raw);

    if (!std.mem.eql(u8, mine, theirs)) return mismatch(
        case,
        "encode",
        P.codec_name,
        "ours != std",
    );

    if (!std.mem.eql(u8, mine, fixture)) return mismatch(
        case,
        "encode",
        P.codec_name,
        "ours != fixture",
    );
}

fn verifyDecode(
    comptime P: type,
    case: []const u8,
    raw: []const u8,
    encoded: []const u8,
    mine: []u8,
    theirs: []u8,
) !void {
    @memset(mine, 0xaa);
    @memset(theirs, 0x55);

    P.mineDecode(mine, encoded);
    P.stdDecode(theirs, encoded);

    if (!std.mem.eql(u8, mine, raw)) return mismatch(
        case,
        "decode",
        P.codec_name,
        "ours != payload",
    );

    if (!std.mem.eql(u8, theirs, raw)) return mismatch(
        case,
        "decode",
        P.codec_name,
        "std != payload",
    );
}

fn verifyDecodeIgnore(
    case: []const u8,
    raw: []const u8,
    wrapped: []const u8,
    mine: []u8,
    theirs: []u8,
) !void {
    const n = mine_ignore.decode(mine, wrapped) catch return mismatch(
        case,
        "decode_ignore",
        "standard",
        "ours errored",
    );

    const m = std_ignore.decode(theirs, wrapped) catch return mismatch(
        case,
        "decode_ignore",
        "standard",
        "std errored",
    );

    if (!std.mem.eql(u8, mine[0..n], raw)) return mismatch(
        case,
        "decode_ignore",
        "standard",
        "ours != payload",
    );

    if (!std.mem.eql(u8, theirs[0..m], raw)) return mismatch(
        case,
        "decode_ignore",
        "standard",
        "std != payload",
    );
}

/// Returns the dest lenght to time with, or null if the input is rejected
/// before decoding starts (`calcSizeForSlice` error), which isn't worth timing.
fn verifyDecodeError(
    comptime P: type,
    case: []const u8,
    bad: []const u8,
    mine_buf: []u8,
    std_buf: []u8,
) !?usize {
    const D = base64.standard.Decoder;
    const S = std.base64.standard.Decoder;

    const len = S.calcSizeForSlice(bad) catch |want| {
        if (D.calcSizeForSlice(bad)) |_| {
            return mismatch(case, "decode_err", P.codec_name, "calcSizeForSlice: ours ok, std error");
        } else |e| if (e != want) return mismatch(case, "decode_err", P.codec_name, "calcSizeForSlice: different error");
        return null;
    };

    const want: ?anyerror = if (S.decode(std_buf[0..len], bad)) |_| null else |e| e;
    const got: ?anyerror = if (D.decode(mine_buf[0..len], bad)) |_| null else |e| e;

    if (want == null) return mismatch(case, "decode_err", P.codec_name, "std accepted the corrupted input");
    if (got != want) return mismatch(case, "decode_err", P.codec_name, "different error than std");

    return len;
}

noinline fn timeRun(
    comptime k: Kernel,
    io: Io,
    out: []u8,
    in: []const u8,
    iterations: usize,
) u64 {
    const start = Io.Clock.now(.awake, io);
    for (0..iterations) |_| {
        // Memory clobber: the input "may have changed", so calls can't be hoisted or merged.
        std.mem.doNotOptimizeAway(in.ptr);
        k(out, in);
    }
    const end = Io.Clock.now(.awake, io);
    return @intCast(@max(0, start.durationTo(end).toNanoseconds()));
}

fn measure(
    comptime mine: Kernel,
    comptime theirs: Kernel,
    env: *const Env,
    mine_out: []u8,
    std_out: []u8,
    in: []const u8,
) Result {
    const io = env.io;
    const target = env.opts.target_ns;

    // Calibrate on ours (double as warm-up), then warm std once.
    var iterations: usize = 1;
    while (true) {
        const t = timeRun(mine, io, mine_out, in, iterations);
        if (t >= target / 8 or iterations >= 1 << 30) {
            const per = @max(1.0, @as(f64, @floatFromInt(t)) / @as(f64, @floatFromInt(iterations)));
            iterations = @max(1, @as(usize, @intFromFloat(@as(f64, @floatFromInt(target)) / per)));
            break;
        }
        iterations *= 2;
    }

    _ = timeRun(theirs, io, std_out, in, @max(1, iterations / 4));

    var mine_ns: [max_runs]f64 = undefined;
    var std_ns: [max_runs]f64 = undefined;

    const runs = env.opts.runs;
    const iter_f: f64 = @floatFromInt(iterations);

    for (0..runs) |r| {
        if (r % 2 == 0) {
            mine_ns[r] = @as(f64, @floatFromInt(timeRun(mine, io, mine_out, in, iterations))) / iter_f;
            std_ns[r] = @as(f64, @floatFromInt(timeRun(theirs, io, std_out, in, iterations))) / iter_f;
        } else {
            std_ns[r] = @as(f64, @floatFromInt(timeRun(theirs, io, std_out, in, iterations))) / iter_f;
            mine_ns[r] = @as(f64, @floatFromInt(timeRun(mine, io, mine_out, in, iterations))) / iter_f;
        }
    }

    return .{
        .iterations = iterations,
        .mine = stats(mine_ns[0..runs]),
        .std = stats(std_ns[0..runs]),
    };
}

fn stats(samples: []f64) Stats {
    std.mem.sort(f64, samples, {}, std.sort.asc(f64));
    const mid = samples.len / 2;
    const median = if (samples.len % 2 == 1)
        samples[mid]
    else
        (samples[mid - 1] + samples[mid]) / 2;

    return .{ .median_ns = median, .min_ns = samples[0] };
}

fn gibPerSecond(bytes: usize, ns: f64) f64 {
    if (ns <= 0 or bytes == 0) return 0;
    return @as(f64, @floatFromInt(bytes)) / ns * 1e9 / (1024.0 * 1024.0 * 1024.0);
}

fn report(
    env: *Env,
    case: []const u8,
    payload_len: usize,
    input_len: usize,
    op: []const u8,
    codec: []const u8,
    r: Result,
) !void {
    const mine_gib = gibPerSecond(payload_len, r.mine.median_ns);
    const std_gib = gibPerSecond(payload_len, r.std.median_ns);
    const speedup = if (r.mine.median_ns > 0) r.std.median_ns / r.mine.median_ns else 0;

    if (env.opts.csv) {
        try env.out.print("{t},{s},{s},{t},{s},{d},{d},{s},{s},{d},{d},{d:.3},{d:.3},{d:.3},{d:.3},{d:.4},{d:.4},{d:.3}\n", .{
            base64.tier,   builtin.cpu.model.name, env.host_cpu,  builtin.mode,
            case,          payload_len,            input_len,     op,
            codec,         r.iterations,           env.opts.runs, r.mine.median_ns,
            r.mine.min_ns, r.std.median_ns,        r.std.min_ns,  mine_gib,
            std_gib,       speedup,
        });
    } else {
        try env.out.print("{s:<20} {s:<16} {s:<16} {d:>10} {d:>12.2} {d:>12.2} {d:>9.3} {d:>9.3} {d:>7.2}x\n", .{
            case,             op,              codec,    payload_len,
            r.mine.median_ns, r.std.median_ns, mine_gib, std_gib,
            speedup,
        });
    }
}
