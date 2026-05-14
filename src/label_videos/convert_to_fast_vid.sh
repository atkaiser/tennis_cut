# Labeling raw videos is very slow, this script converts the video
# to a format that makes it fast to label

for f in *.MOV; do
  [ -e "$f" ] || continue
  echo "Processing $f"

  fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
        -of default=noprint_wrappers=1:nokey=1 "$f")

  if [ "$fps" = "120/1" ]; then
    echo "$f is already 120fps - checking first 5s keyframes"
    sample_counts=$(ffprobe -v error -select_streams v:0 -read_intervals "%+5" \
          -show_frames -show_entries frame=key_frame -of default=noprint_wrappers=1:nokey=1 "$f" \
          | awk '/^[01]$/ { total++; if ($1 == 1) key++ } END { printf "%d %d", total, key }')
    sample_total=${sample_counts%% *}
    sample_keyframes=${sample_counts##* }

    if [ "$sample_total" -gt 0 ] && [ "$sample_total" = "$sample_keyframes" ]; then
      echo "$f already processed (first 5s all-keyframe) - skipping"
      continue
    fi

    echo "$f is 120fps but first 5s keyframes are $sample_keyframes/$sample_total - reprocessing"
  fi

  echo "$f"
  tmp="${f%.*}.tmp.${f##*.}"
  if ffmpeg -hide_banner -loglevel error -y -i "$f" \
    -vf fps=120 \
    -c:v libx264 -preset veryfast -crf 18 \
    -x264-params keyint=1:min-keyint=1:scenecut=0:bf=0 \
    -movflags +faststart \
    -c:a copy "$tmp" && mv "$tmp" "$f"; then
    :
  fi
done
