This project contains a few subprojects. All of the code is in the src
directory with subdirectories in there each being a subproject. Each subproject
should have it's own README file.

There is the examples directory, which contains data files that are
useful for test runs of scripts. However because some of the video and
audio files are large, they aren't included here and must be downloaded
from the web. From the root directory you can run:

1. wget 'https://www.dropbox.com/scl/fi/dce0wabuy0kss3xtcp7th/tester.MOV?rlkey=y8cwf7wssvswq1dxj12rxrrhw&st=i4yi7aeh&dl=0' -O examples/videos/tester.MOV
2. wget 'https://www.dropbox.com/scl/fi/5pxrm1y9ij8qvls07ve03/tester.wav?rlkey=rxlfhsdxigrqf3zwye6jk8b1q&st=icvmew0l&dl=0' -O examples/wavs/tester.wav

It's not so useful to find accurate results
but can be useful to test if a script runs without any errors. Some examples
of what can be run after downloading the files:

1. Run from inside `examples/videos`: `../../src/label_videos/convert_to_fast_vid.sh`
2. Run from project root: `python src/train_pop_detector/prepare_audio_windows.py --videos_dir examples/videos --wav_dir /tmp/wavs --out_csv examples/meta/labled_windows.csv` (Use /tmp so that it doesn't skip work of previous runs)
3. Run from project root: `python src/train_pop_detector/train_audio_pop.py examples/meta/labled_windows.csv --epochs 3 --device cpu --bs 1`
