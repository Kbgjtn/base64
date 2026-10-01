pub const base64 = @import("base64.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
