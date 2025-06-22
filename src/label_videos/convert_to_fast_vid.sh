# Labeling raw videos is very slow, this script converts the video
# to a format that makes it fast to label

for f in *.MOV; do
  [ -e "$f" ] || continue
  fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
        -of default=noprint_wrappers=1:nokey=1 "$f")
  if [ "$fps" = "120/1" ]; then
    echo "$f already processed – skipping"
    continue
  fi
  echo "$f"
  tmp="${f%.*}.tmp.${f##*.}"
  ffmpeg -hide_banner -loglevel error -y -i "$f" \
    -vf fps=120 \
    -c:v libx264 -preset veryfast -crf 18 \
    -x264-params keyint=1:min-keyint=1:scenecut=0:bf=0 \
    -movflags +faststart \
    -c:a copy "$tmp" && mv "$tmp" "$f"
done
