#!/bin/sh
# Regenerates ll_*.webp and decode_oracle.txt for the decoder tests, with
# cwebp and dwebp 1.6.0 (brew install webp). Run from this directory.
# decode_oracle.txt has "file sha256" of the RGBA pixels dwebp -pam writes.
set -e
here=$(pwd)
root=$(cd ../../.. && pwd)
tmp=$(mktemp -d)
mkdir "$tmp/gen"
cp gen_decode.go.txt "$tmp/gen/main.go"
(cd "$tmp/gen" && go run main.go "$tmp" "$here/kodim05_192.png")
rm -f ll_*.webp lossy_*.webp
for p in "$tmp"/*.png; do
  b=$(basename "$p" .png)
  for z in 0 4 9; do
    cwebp -quiet -lossless -z $z -exact "$p" -o "ll_${b}_z$z.webp"
  done
done
# Lossy: six encoder setups on the photo (default; coarse and fast; fine and
# slow, large tokens; simple filter with sharpness; no filter; strongest filter), the default on
# the other opaque images and on two Kodak crops.
cwebp -quiet -q 75 "$tmp/photo.png" -o lossy_photo_a.webp
cwebp -quiet -q 5 -m 0 -segments 1 "$tmp/photo.png" -o lossy_photo_b.webp
cwebp -quiet -q 100 -m 6 "$tmp/photo.png" -o lossy_photo_c.webp
cwebp -quiet -q 50 -nostrong -f 80 -sharpness 3 "$tmp/photo.png" -o lossy_photo_d.webp
cwebp -quiet -q 60 -f 0 "$tmp/photo.png" -o lossy_photo_e.webp
cwebp -quiet -q 40 -sharpness 7 -f 100 "$tmp/photo.png" -o lossy_photo_f.webp
for b in runs palc pald; do
  cwebp -quiet -q 75 "$tmp/$b.png" -o "lossy_${b}.webp"
done
for k in kodim05 kodim23; do
  cwebp -quiet -q 75 "$k"_192.png -o "lossy_${k}.webp"
done
# Lossy with ALPH: raw, and lossless-coded with each filter choice and with
# quantized levels, on three images and the 1 pixel wide and tall ones.
for b in photoa alpha; do
  cwebp -quiet -q 75 -alpha_method 0 "$tmp/$b.png" -o "lossy_alpha_${b}_raw.webp"
  for flt in none fast best; do
    cwebp -quiet -q 75 -alpha_method 1 -alpha_filter $flt "$tmp/$b.png" -o "lossy_alpha_${b}_$flt.webp"
  done
  cwebp -quiet -q 75 -alpha_q 30 -alpha_filter best "$tmp/$b.png" -o "lossy_alpha_${b}_q30.webp"
done
for b in tiny11 tiny19 tiny91; do
  cwebp -quiet -q 75 -alpha_filter best "$tmp/$b.png" -o "lossy_alpha_$b.webp"
done
# cwebp never picks the vertical alpha filter on these, so filter one raw
# ALPH chunk by hand: each row minus the row above, the first row left to right.
python3 - 97 61 <<'PY'
import sys
w, h = int(sys.argv[1]), int(sys.argv[2])
d = bytearray(open("lossy_alpha_photoa_raw.webp", "rb").read())
i = d.index(b"ALPH") + 8
a = d[i + 1 : i + 1 + w * h]
o = bytearray(a)
for y in range(h):
    for x in range(w):
        p = (a[(y - 1) * w + x] if y else (a[x - 1] if x else 0))
        o[y * w + x] = (a[y * w + x] - p) & 255
d[i] = 2 << 2
d[i + 1 : i + 1 + w * h] = o
open("lossy_alpha_vfilter.webp", "wb").write(d)
PY
# This repo's encoder writes two token partitions and skip flags.
mkdir "$root/.gentmp"
cp gen_encode.go.txt "$root/.gentmp/main.go"
(cd "$root" && go run ./.gentmp "$tmp/photo.png" "$here/lossy_ours_photo.webp")
rm -rf "$root/.gentmp"
: > decode_oracle.txt
# he_768.webp is the encoder tests' lossless fixture.
for f in ll_*.webp he_768.webp lossy_*.webp; do
  dwebp -quiet -nodither -pam "$f" -o "$tmp/o.pam"
  # the pixels are the last width*height*4 bytes of the PAM
  w=$(head -n 3 "$tmp/o.pam" | awk '$1 == "WIDTH" {print $2}')
  h=$(head -n 3 "$tmp/o.pam" | awk '$1 == "HEIGHT" {print $2}')
  s=$(tail -c $((w * h * 4)) "$tmp/o.pam" | shasum -a 256 | cut -d' ' -f1)
  echo "$f $s" >> decode_oracle.txt
done
rm -rf "$tmp"
