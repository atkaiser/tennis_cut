# Labeling raw videos is very slow, this script converts the video
# to a format that makes it fast to label

for f in `ls *.MOV`; do echo $f ; tmp="${f%.*}.tmp.${f##*.}"; \
ffmpeg -hide_banner -loglevel error -y -i "$f" \
  -vf fps=120 \
  -c:v libx264 -preset veryfast -crf 18 \
  -x264-params keyint=1:min-keyint=1:scenecut=0:bf=0 \
  -movflags +faststart \
  -c:a copy "$tmp" && mv "$tmp" "$f"; done