#!/bin/sh
# Regenerates anim_*.webp and anim_oracle.txt with cwebp, webpmux and dwebp
# 1.6.0 (brew install webp). Run from this directory. anim_oracle.txt has
# "file frames_dropped sha256": the RGBA of frame 1 as dwebp -pam decodes it,
# placed at its ANMF offset on a transparent canvas, as WebPAnimDecoder does.
set -e
tmp=$(mktemp -d)
python3 - "$tmp/a.png" <<'PY'
import sys, struct, zlib
w, h = 40, 24
raw = b"".join(b"\0" + b"".join(bytes([x * 6 & 255, y * 10 & 255, 90, 255 if (x // 4 + y // 4) % 3 else 40]) for x in range(w)) for y in range(h))
def ch(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
open(sys.argv[1], "wb").write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b""))
PY
k=kodim05_192.png
cwebp -quiet -lossless -exact -crop 10 10 40 24 $k -o "$tmp/l1.webp"
cwebp -quiet -lossless -crop 60 60 64 48 $k -o "$tmp/l2.webp"
cwebp -quiet -q 75 -crop 20 20 48 32 $k -o "$tmp/m1.webp"
cwebp -quiet -q 75 -crop 70 20 48 32 $k -o "$tmp/m2.webp"
cwebp -quiet -q 75 -crop 20 70 48 32 $k -o "$tmp/m3.webp"
cwebp -quiet -q 75 -alpha_method 1 "$tmp/a.png" -o "$tmp/a1.webp"
cwebp -quiet -q 75 -crop 30 30 64 40 $k -o "$tmp/a2.webp"
webpmux -frame "$tmp/l1.webp" +100+6+2+1+b -frame "$tmp/l2.webp" +100+0+0+1+b -o anim_lossless_off.webp >/dev/null
webpmux -frame "$tmp/m1.webp" +100+0+0+1+b -frame "$tmp/m2.webp" +100+0+0+1+b -frame "$tmp/m3.webp" +100+0+0+1+b -o anim_lossy_full.webp >/dev/null
webpmux -frame "$tmp/a1.webp" +100+10+4+1+b -frame "$tmp/a2.webp" +100+0+0+1+b -o anim_alpha_off.webp >/dev/null
: > anim_oracle.txt
for f in anim_*.webp; do
  webpmux -get frame 1 "$f" -o "$tmp/f.webp"
  dwebp -quiet -nodither -pam "$tmp/f.webp" -o "$tmp/f.pam"
  python3 - "$f" "$tmp/f.pam" >> anim_oracle.txt <<'PY'
import sys, hashlib, struct, re
d = open(sys.argv[1], "rb").read()
cw, ch = [1 + int.from_bytes(d[24 + o:27 + o], "little") for o in (0, 3)]
i = d.index(b"ANMF")
x, y = [2 * int.from_bytes(d[i + 8 + o:i + 11 + o], "little") for o in (0, 3)]
n = d.count(b"ANMF")
p = open(sys.argv[2], "rb").read()
m = re.match(rb"P7\nWIDTH (\d+)\nHEIGHT (\d+)\n", p)
w, h = int(m[1]), int(m[2])
px = p[-4 * w * h:]
c = bytearray(4 * cw * ch)
for r in range(h):
    c[4 * ((y + r) * cw + x):4 * ((y + r) * cw + x + w)] = px[4 * r * w:4 * (r + 1) * w]
print(sys.argv[1], n - 1, hashlib.sha256(c).hexdigest())
PY
done
rm -rf "$tmp"
