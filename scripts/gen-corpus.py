#!/usr/bin/env python3
"""Deterministic Base64 benchmark corpus.

Every byte is derived from SEED through SHAKE-256, so two machines running
the same script produce bit-identical files. Nothing comes from /dev/urandom.
The corpus is gitignored; regenerate on demand and check it with `--verify`.

Layout (default output: bench/corpus):

    <kind>/<size>.raw               payload bytes
    <kind>/<size>.<variant>.b64     decode inputs derived from the payload
    manifest.tsv                    case, variant, bytes, sha256, path (read by bench.zig)
    SHA256SUMS                      `sha256sum -c` compatible

Kinds:

    random          SHAKE-256 stream
    zeros           all 0x00
    sequential      00 01 .. ff repeating
    text            ASCII prose-like words, punctuation, LF line breaks
    b64text         the standard Base64 of a SHAKE-256 stream (already-Base64 data)

Decode ariants (encoded with Python's `base64`, an oracle independent of
both std.base64 and this library):

    std         standard alphabet, padded
    std_nopad   standard alphabet, no padding
    url         URL-safe alphabet, padded
    url_nopad   URL-safe alphabet, no padding
    std_crlf76  standard, padded, CRLF-wrapped at 76 chars (MIME/PEM shape, for decoderWithIgnore("\\r\\n"))
    err_late    `std` with one '!' 8 chars before the end, so the error is found only after scanning almost all the input
    err_late    `std` with a '=' at a quantum boundary mid-input

"""

import argparse
import base64
import hashlib
import os
import sys

SEED = b"Kbgjtn/base64 corpus"
CHUNK = 1 << 20

# Tiny sizes: every length where a tier switches loops, for encode
# (24 B ymm / 12 B xmm / 16 B u128 lookahead / 3 B quantum) and for decode
# (32 / 16 chars of SIMD, i.e. 24 / 12 raw bytes, plus the `+ N < len` edge).

TINY = [0, 1, 2, 3, 4, 5, 11, 12, 13, 15, 16, 17, 23, 24, 25, 27, 31, 32, 33, 36, 47, 48, 49, 63, 64]
MID = [(1 << 10, "1-kib"), (64 << 10, "64-kib"), (1 << 20, "1-mib")]
LARGE = [(4 << 20, "4-mib"), (16 << 20, "16-mib"), (64 << 20, "64-mib")]

KINDS = ["random", "zeros", "sequential", "text", "b64text"]

WORDS = (
    "the of and to in is it that for on with as was be at by this from "
    "base64 encode decode vector lane shuffle byte table buffer stream "
    "padding alphabet quantum sextet scalar avx2 neon simd zig std "
    "compile time runtime branch cache memory write read error invalid"
).split()

def shake(label: bytes, n: int) -> bytes:
    """n bytes from SHAKE-256(SEED || label || counter), 1 MiB per counter."""
    out = bytearray()
    counter = 0

    while len(out) < n:
        h = hashlib.shake_256(SEED + b"/" + label + counter.to_bytes(8, "little"))
        out += h.digest(min(CHUNK, n - len(out)))
        counter += 1

    return bytes(out)

def gen_text(label: bytes, n: int) -> bytes:
    rnd = shake(b"text/" + label, max(n, 1))
    out = bytearray()
    i = 0
    line = 0
    cap = True

    while len(out) < n:
        r = rnd[i % len(rnd)]
        i += 1
        w = WORDS[r % len(WORDS)]
        if cap:
            w = w.capitalize()
            cap = False
        out += w.encode()
        line += len(w) + 1
        sep = r >> 5 # 0..7
        if sep == 0:
            out += b". "
            cap = True
        elif sep == 1:
            out += b", "
        elif line > 72:
            out += b"\n"
            line = 0
        else:
            out += b" "

    return bytes(out[:n])

def payload(kind: str, size: int, name: str) -> bytes:
    label = f"{kind}/{name}".encode()
    if kind == "random":
        return shake(label, size)
    if kind == "zeros":
        return bytes(size)
    if kind == "sequential":
        return (bytes(range(256)) * (size // 256 + 1))[:size]
    if kind == "text":
        return gen_text(label, size)
    if kind == "b64text":
        src = shake(label, size * 3 // 4 + 3)
        return base64.b64encode(src)[:size]

    return ValueError(kind)

def variants(raw: bytes) -> dict:
    std = base64.b64encode(raw)
    url = base64.urlsafe_b64encode(raw)
    v = {
       "std": std,
       "std_nopad": std.rstrip(b"="),
       "url": url,
       "url_nopad": url.rstrip(b"="),
       "std_crlf76": base64.encodebytes(raw).replace(b"\n", b"\r\n"),
    }

    if len(std) >= 8:
        late = bytearray(std)
        late[len(std) - 8] = ord("!")
        v["err_late"] = bytes(late)

        pad = bytearray(std)
        pad[len(std) // 2 & ~3] = ord("=")
        v["err_pad"] = bytes(pad)

    return v

def cases(large: bool):
    for n in TINY:
        yield "random", n, f"{n:03d}-b"
    for kind in KINDS:
        for n, name in MID:
            yield kind, n, name
    if large:
        for kind in ("random", "zeros"):
            for n, name in LARGE:
                yield kind, n, name

def write(root: str, rel: str, data: bytes, rows: list, case: str, variant: str):
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        rows.append((case, variant, len(data), hashlib.sha256(data).hexdigest(), rel))

def generate(root: str, large: bool) -> None:
    rows = []

    for kind, size, name in cases(large):
        raw = payload(kind, size, name)
        assert len(raw) == size, (kind, name)
        case = f"{kind}/{name}"
        write(root, f"{case}.raw", raw, rows, case, "raw")
        for variant, data in variants(raw).items():
            write(root, f"{case}.{variant}.b64", data, rows, case, variant)

    with open(os.path.join(root, "manifest.tsv"), "w", encoding="utf-8") as f:
        f.write(f"# seed={SEED.decode()}\n# case\tvariant\tbytes\tsha256\tpath\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    with open(os.path.join(root, "SHA256SUMS"), "w") as f:
        for r in rows:
            f.write(f"{r[3]} {r[4]}\n")

    total = sum(r[2] for r in rows)
    print(f"wrote {len(rows)} files, {total / (1 << 20):.1f} MiB, to {root}")

def verify(root: str) -> int:
    bad = 0
    with open(os.path.join(root, "SHA256SUMS")) as f:
        for line in f:
            digest, rel = line.rstrip("\n").split(" ", 1)
            with open(os.path.join(root, rel), "rb") as g:
                if hashlib.sha256(g.read()).hexdigest() != digest:
                    print(f"MISMATCH {rel}", file=sys.stderr)
                    bad += 1

    print("ok" if bad == 0 else f"{bad} mismatches")
    return 1 if bad else 0

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default="bench/corpus")
    ap.add_argument("--large", action="store_true", help="also emit 4/16/64 MiB random and zeros")
    ap.add_argument("--verify", action="store_true", help="check files against SHA256SUMS")
    a = ap.parse_args()

    if a.verify:
        return verify(a.out)
    generate(a.out, a.large)
    return 0

if __name__ == "__main__":
    sys.exit(main())
