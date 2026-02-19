# Convert in-place: videos/foo.MOV -> videos/foo.mp4
for f in videos/*.MOV; do
  [ -e "$f" ] || continue
  echo "$f"

  out="${f%.*}.mp4"
  tmp="${out}.tmp.mp4"

  ffmpeg -hide_banner -y \
    -hwaccel videotoolbox -i "$f" \
    -vf "fps=120,scale=-2:1080" \
    -c:v hevc_videotoolbox -q:v 35 -tag:v hvc1 \
    -c:a copy \
    "$tmp" && mv -f "$tmp" "$out" && rm -f "$f"
done