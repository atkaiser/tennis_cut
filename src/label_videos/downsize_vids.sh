# Command used to convert video files to 1080p and 120 fps
for f in `ls videos/*.MOV`; do \
  echo $f
  ffmpeg -hide_banner -y \
    -hwaccel videotoolbox -i "$f" \
    -vf "fps=120,scale=-2:1080" \
    -c:v hevc_videotoolbox -q:v 35 -tag:v hvc1 \
    -c:a copy \
    "${f%.*}_1080p120.mp4"; \
done