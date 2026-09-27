const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const custom_base64 = @import("base64");
const StdEncoder = std.base64.standard.Encoder;

const TestCase = struct {
    name: []const u8,
    data: []const u8,
};

const small_test_cases = [_]TestCase{
    // Random

    .{
        .name = "random/empty",
        .data = @embedFile("testdata/random/empty.bin"),
    },
    .{
        .name = "random/1 byte",
        .data = @embedFile("testdata/random/1-byte.bin"),
    },
    .{
        .name = "random/2 bytes",
        .data = @embedFile("testdata/random/2-bytes.bin"),
    },
    .{
        .name = "random/3 bytes",
        .data = @embedFile("testdata/random/3-bytes.bin"),
    },
    .{
        .name = "random/4 bytes",
        .data = @embedFile("testdata/random/4-bytes.bin"),
    },
    .{
        .name = "random/11 bytes",
        .data = @embedFile("testdata/random/11-bytes.bin"),
    },
    .{
        .name = "random/12 bytes",
        .data = @embedFile("testdata/random/12-bytes.bin"),
    },
    .{
        .name = "random/15 bytes",
        .data = @embedFile("testdata/random/15-bytes.bin"),
    },
    .{
        .name = "random/16 bytes",
        .data = @embedFile("testdata/random/16-bytes.bin"),
    },
    .{
        .name = "random/17 bytes",
        .data = @embedFile("testdata/random/17-bytes.bin"),
    },

    // Block/alignment boundaries.
    .{
        .name = "random/31 bytes",
        .data = @embedFile("testdata/random/31-bytes.bin"),
    },
    .{
        .name = "random/32 bytes",
        .data = @embedFile("testdata/random/32-bytes.bin"),
    },
    .{
        .name = "random/33 bytes",
        .data = @embedFile("testdata/random/33-bytes.bin"),
    },
    .{
        .name = "random/63 bytes",
        .data = @embedFile("testdata/random/63-bytes.bin"),
    },
    .{
        .name = "random/64 bytes",
        .data = @embedFile("testdata/random/64-bytes.bin"),
    },
    .{
        .name = "random/65 bytes",
        .data = @embedFile("testdata/random/65-bytes.bin"),
    },
    .{
        .name = "random/127 bytes",
        .data = @embedFile("testdata/random/127-bytes.bin"),
    },
    .{
        .name = "random/128 bytes",
        .data = @embedFile("testdata/random/128-bytes.bin"),
    },
    .{
        .name = "random/129 bytes",
        .data = @embedFile("testdata/random/129-bytes.bin"),
    },
    .{
        .name = "random/255 bytes",
        .data = @embedFile("testdata/random/255-bytes.bin"),
    },
    .{
        .name = "random/256 bytes",
        .data = @embedFile("testdata/random/256-bytes.bin"),
    },
    .{
        .name = "random/257 bytes",
        .data = @embedFile("testdata/random/257-bytes.bin"),
    },

    // Throughput / working-set sizes.
    .{
        .name = "random/1 KiB",
        .data = @embedFile("testdata/random/1-kib.bin"),
    },
    .{
        .name = "random/4 KiB",
        .data = @embedFile("testdata/random/4-kib.bin"),
    },
    .{
        .name = "random/16 KiB",
        .data = @embedFile("testdata/random/16-kib.bin"),
    },
    .{
        .name = "random/64 KiB",
        .data = @embedFile("testdata/random/64-kib.bin"),
    },
    .{
        .name = "random/256 KiB",
        .data = @embedFile("testdata/random/256-kib.bin"),
    },

    // Zeros

    .{
        .name = "zeros/1 KiB",
        .data = @embedFile("testdata/zeros/1-kib.bin"),
    },
    .{
        .name = "zeros/4 KiB",
        .data = @embedFile("testdata/zeros/4-kib.bin"),
    },
    .{
        .name = "zeros/16 KiB",
        .data = @embedFile("testdata/zeros/16-kib.bin"),
    },
    .{
        .name = "zeros/64 KiB",
        .data = @embedFile("testdata/zeros/64-kib.bin"),
    },
    .{
        .name = "zeros/256 KiB",
        .data = @embedFile("testdata/zeros/256-kib.bin"),
    },

    // Sequential

    .{
        .name = "sequential/1 KiB",
        .data = @embedFile("testdata/sequential/1-kib.bin"),
    },
    .{
        .name = "sequential/4 KiB",
        .data = @embedFile("testdata/sequential/4-kib.bin"),
    },
    .{
        .name = "sequential/16 KiB",
        .data = @embedFile("testdata/sequential/16-kib.bin"),
    },
    .{
        .name = "sequential/64 KiB",
        .data = @embedFile("testdata/sequential/64-kib.bin"),
    },
    .{
        .name = "sequential/256 KiB",
        .data = @embedFile("testdata/sequential/256-kib.bin"),
    },
};

const large_test_cases = if (build_options.include_large) [_]TestCase{
    .{
        .name = "random/1 MiB",
        .data = @embedFile("testdata/random/1-mib.bin"),
    },
    .{
        .name = "random/4 MiB",
        .data = @embedFile("testdata/random/4-mib.bin"),
    },
    .{
        .name = "random/16 MiB",
        .data = @embedFile("testdata/random/16-mib.bin"),
    },
    .{
        .name = "random/32 MiB",
        .data = @embedFile("testdata/random/32-mib.bin"),
    },
    .{
        .name = "random/64 MiB",
        .data = @embedFile("testdata/random/64-mib.bin"),
    },
    .{
        .name = "random/128 MiB",
        .data = @embedFile("testdata/random/128-mib.bin"),
    },
    .{
        .name = "random/256 MiB",
        .data = @embedFile("testdata/random/256-mib.bin"),
    },

    .{
        .name = "zeros/1 MiB",
        .data = @embedFile("testdata/zeros/1-mib.bin"),
    },
    .{
        .name = "zeros/4 MiB",
        .data = @embedFile("testdata/zeros/4-mib.bin"),
    },
    .{
        .name = "zeros/16 MiB",
        .data = @embedFile("testdata/zeros/16-mib.bin"),
    },
    .{
        .name = "zeros/32 MiB",
        .data = @embedFile("testdata/zeros/32-mib.bin"),
    },
    .{
        .name = "zeros/64 MiB",
        .data = @embedFile("testdata/zeros/64-mib.bin"),
    },
    .{
        .name = "zeros/128 MiB",
        .data = @embedFile("testdata/zeros/128-mib.bin"),
    },
    .{
        .name = "zeros/256 MiB",
        .data = @embedFile("testdata/zeros/256-mib.bin"),
    },

    .{
        .name = "sequential/1 MiB",
        .data = @embedFile("testdata/sequential/1-mib.bin"),
    },
    .{
        .name = "sequential/4 MiB",
        .data = @embedFile("testdata/sequential/4-mib.bin"),
    },
    .{
        .name = "sequential/16 MiB",
        .data = @embedFile("testdata/sequential/16-mib.bin"),
    },
    .{
        .name = "sequential/32 MiB",
        .data = @embedFile("testdata/sequential/32-mib.bin"),
    },
    .{
        .name = "sequential/64 MiB",
        .data = @embedFile("testdata/sequential/64-mib.bin"),
    },
    .{
        .name = "sequential/128 MiB",
        .data = @embedFile("testdata/sequential/128-mib.bin"),
    },
    .{
        .name = "sequential/256 MiB",
        .data = @embedFile("testdata/sequential/256-mib.bin"),
    },
} else [_]TestCase{};

const test_cases = small_test_cases ++ large_test_cases;

const warmup_iterations: usize = 100;

/// Target bytes processed per case once inputs are large enough that per-call overhead
/// no longer dominates timing (see the fixed tiers below for sizes <=1 MiB, which are
/// hand-tuned for exactly that reason).
///
/// 200 MiB keeps a 256 MiB case down to a handful of iterations instead of the ~25 GiB
/// the old flat `100` would have processed at that size.
const large_input_byte_budget: usize = 200 * mib;

/// Floor so even the largest embedded case still gets enough samples for
/// a non-degenerate timing — below this, one scheduling hiccup dominates
/// the whole measurement.
const min_large_iterations: usize = 3;

fn iterationsFor(input_len: usize) usize {
    return switch (input_len) {
        0...64 => 1_000_000,
        65...4 * kib => 100_000,
        4 * kib + 1...64 * kib => 10_000,
        64 * kib + 1...mib => 1_000,
        else => @max(
            min_large_iterations,
            @min(100, large_input_byte_budget / input_len),
        ),
    };
}

fn warmupIterationsFor(iterations: usize) usize {
    return @min(warmup_iterations, iterations);
}

const max_input_len: usize = blk: {
    var result: usize = 0;

    for (test_cases) |test_case| {
        if (test_case.data.len > result) {
            result = test_case.data.len;
        }
    }

    break :blk result;
};

const max_output_len = StdEncoder.calcSize(max_input_len);

const kib = 1 << 10;
const mib = kib << 10;

const Codec = enum {
    custom,
    std,
};

const BenchmarkResult = struct {
    total_ns: u64,
    checksum: u64,

    fn nsPerOperation(self: BenchmarkResult, iterations: usize) u64 {
        return self.total_ns / iterations;
    }

    fn mibPerSecond(
        self: BenchmarkResult,
        input_len: usize,
        iterations: usize,
    ) u128 {
        if (self.total_ns == 0) return 0;

        const bytes_processed =
            @as(u128, input_len) * @as(u128, iterations);

        return (bytes_processed * std.time.ns_per_s) /
            @as(u128, self.total_ns) /
            (1024 * 1024);
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    const custom_output_storage = try gpa.alloc(u8, max_output_len);
    defer gpa.free(custom_output_storage);

    const std_output_storage = try gpa.alloc(u8, max_output_len);
    defer gpa.free(std_output_storage);

    try stdout.print(
        "Base64 benchmark\n" ++
            "Build with: zig build -Doptimize={s} -Dcpu={s} bench.zig\n\n",
        .{
            @tagName(builtin.mode),
            builtin.cpu.model.name,
        },
    );
    try stdout.print(
        "{s:<14} {s:<8} {s:>12} {s:>14} {s:>14} {s:>14} {s:>14} {s:>10}\n",
        .{ "input", "codec", "iterations", "buffer ns/op", "pool ns/op", "alloc ns/op", "buffer MiB/s", "speedup" },
    );
    try stdout.print(
        "{s:-<14} {s:-<8} {s:->12} {s:->14} {s:->14} {s:->14} {s:->14} {s:->10}\n",
        .{ "", "", "", "", "", "", "", "" },
    );

    if (!build_options.include_large) {
        try stdout.print(
            "(large testdata not embedded; rebuild with -Dlarge=true to include it)\n\n",
            .{},
        );
    }

    for (test_cases) |case| {
        try runCase(io, gpa, stdout, case, custom_output_storage, std_output_storage);
    }
}

fn runCase(
    io: std.Io,
    gpa: std.mem.Allocator,
    stdout: *std.Io.Writer,
    case: TestCase,
    custom_output_storage: []u8,
    std_output_storage: []u8,
) !void {
    const input = case.data;
    const input_len = input.len;

    const iterations = iterationsFor(input_len);

    const custom_size = custom_base64.computeSize(input_len);
    const std_size = StdEncoder.calcSize(input_len);
    if (custom_size != std_size) return error.SizeMismatch;

    const custom_output = custom_output_storage[0..custom_size];
    const std_output = std_output_storage[0..std_size];

    try verifyEncoding(input, custom_output, std_output);

    var pool = std.heap.ArenaAllocator.init(gpa);
    defer pool.deinit();

    const custom_buffer = try benchmark(customEncode, io, input, custom_size, iterations, .{ .buffer = custom_output });
    const custom_pool = try benchmark(customEncode, io, input, custom_size, iterations, .{ .pool = &pool });
    const custom_alloc = try benchmark(customEncode, io, input, custom_size, iterations, .{ .alloc = gpa });

    const std_buffer = try benchmark(stdEncode, io, input, std_size, iterations, .{ .buffer = std_output });
    const std_pool = try benchmark(stdEncode, io, input, std_size, iterations, .{ .pool = &pool });
    const std_alloc = try benchmark(stdEncode, io, input, std_size, iterations, .{ .alloc = gpa });

    try printResult(stdout, case.name, .custom, input_len, iterations, custom_buffer, custom_pool, custom_alloc, null);
    try printResult(stdout, case.name, .std, input_len, iterations, std_buffer, std_pool, std_alloc, custom_buffer);

    try stdout.flush();
}

fn verifyEncoding(
    input: []const u8,
    custom_output: []u8,
    std_output: []u8,
) !void {
    const custom_len = encodeInto(
        .custom,
        custom_output,
        input,
    );

    const std_len = encodeInto(
        .std,
        std_output,
        input,
    );

    if (custom_len != std_len) {
        return error.EncodedLengthMismatch;
    }

    if (!std.mem.eql(
        u8,
        custom_output[0..custom_len],
        std_output[0..std_len],
    )) {
        return error.EncodedOutputMismatch;
    }
}

fn encodeInto(
    comptime codec: Codec,
    output: []u8,
    input: []const u8,
) usize {
    return switch (codec) {
        .custom => custom_base64.Encoder.encodeToSlice(output, input),
        .std => StdEncoder.encode(output, input).len,
    };
}

fn expectedSize(
    comptime codec: Codec,
    input_len: usize,
) usize {
    return switch (codec) {
        .custom => custom_base64.computeSize(input_len),
        .std => StdEncoder.calcSize(input_len),
    };
}

const OutputMode = union(enum) {
    buffer: []u8,
    alloc: std.mem.Allocator,
    /// A reused arena, reset (not freed) after every iteration via `.retain_capacity`.
    /// Arena allocation is a bump-pointer op and the reset is O(1), so this isolates
    /// "pay for a fresh buffer and encode" from a general-purpose allocator's malloc/free/mmap
    /// bookkeeping. If pool ns/op lands close to buffer ns/op, the alloc/buffer gap really is
    /// the allocator, not the codec's own allocation path.
    pool: *std.heap.ArenaAllocator,
};

const Op = fn (output: []u8, input: []const u8) anyerror!usize;

noinline fn benchmark(
    comptime run: Op,
    io: std.Io,
    input: []const u8,
    expected_len: usize,
    iterations: usize,
    mode: OutputMode,
) !BenchmarkResult {
    var checksum: u64 = 0;

    const step = struct {
        fn call(m: OutputMode, in: []const u8, exp_len: usize, sum: *u64) !void {
            const output = switch (m) {
                .buffer => |b| b,
                .alloc => |a| try a.alloc(u8, exp_len),
                .pool => |arena| try arena.allocator().alloc(u8, exp_len),
            };

            defer switch (m) {
                .buffer => {},
                .alloc => |a| a.free(output),
                .pool => |arena| _ = arena.reset(.retain_capacity),
            };

            const encoded_len = try run(output, in);
            if (encoded_len != exp_len) return error.LengthMismatch;
            if (encoded_len != 0) {
                sum.* +%= output[0];
                sum.* +%= output[encoded_len - 1];
            }
        }
    }.call;

    // Warm up the implementation, allocator, and instruction/data caches.
    for (0..warmupIterationsFor(iterations)) |_| try step(mode, input, expected_len, &checksum);

    const start = std.Io.Clock.now(.awake, io);
    for (0..iterations) |_| try step(mode, input, expected_len, &checksum);
    const end = std.Io.Clock.now(.awake, io);

    std.mem.doNotOptimizeAway(checksum);

    return .{
        .total_ns = @intCast(start.durationTo(end).toNanoseconds()),
        .checksum = checksum,
    };
}
fn customEncode(output: []u8, input: []const u8) !usize {
    return custom_base64.Encoder.encodeToSlice(output, input);
}
fn stdEncode(output: []u8, input: []const u8) !usize {
    return StdEncoder.encode(output, input).len;
}

fn printResult(
    stdout: *std.Io.Writer,
    name: []const u8,
    codec: Codec,
    input_len: usize,
    iterations: usize,
    buffer: BenchmarkResult,
    pool: BenchmarkResult,
    alloc: BenchmarkResult,
    comparison: ?BenchmarkResult,
) !void {
    const speedup = if (comparison) |reference|
        if (buffer.total_ns == 0)
            0.0
        else
            @as(f64, @floatFromInt(reference.total_ns)) /
                @as(f64, @floatFromInt(buffer.total_ns))
    else
        1.0;

    try stdout.print(
        "{s:<14} {t:<8} {d:>12} {d:>14} {d:>14} {d:>14} {d:>14} {d:>9.2}x\n",
        .{
            name,
            codec,
            iterations,
            buffer.nsPerOperation(iterations),
            pool.nsPerOperation(iterations),
            alloc.nsPerOperation(iterations),
            buffer.mibPerSecond(input_len, iterations),
            speedup,
        },
    );
}
