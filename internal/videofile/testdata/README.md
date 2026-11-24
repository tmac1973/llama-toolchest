# Test clips

Made once with ffmpeg n9.0.2. The app itself never runs ffmpeg.

```sh
F() { ffmpeg -hide_banner -loglevel error -y "$@"; }
V="-f lavfi -i testsrc=size=64x48:rate=16"
A="-f lavfi -i sine=frequency=440:sample_rate=16000"
F $V $A -frames:v 8 -shortest -c:v mjpeg -q:v 5 -c:a pcm_s16le mjpeg-pcm-64x48-8f.avi
F $V $A -frames:v 4 -shortest -c:v mjpeg -q:v 5 -c:a aac -b:a 32k mjpeg-aac.avi
F -f lavfi -i testsrc=size=64x48:rate=10 -frames:v 6 -loop 0 anim-64x48-6f.webp
F -f lavfi -i testsrc=size=64x48 -frames:v 1 still-64x48.webp
F $V -frames:v 9 -c:v libvpx -g 1 -auto-alt-ref 0 -b:v 200k vp8-key-64x48-9f.webm
F $V $A -frames:v 9 -shortest -c:v libvpx -g 1 -auto-alt-ref 0 -b:v 200k -c:a pcm_s16le -f matroska vp8-key-pcm.mkv
F $V -f lavfi -i sine=frequency=440:sample_rate=48000 -frames:v 9 -shortest -c:v libvpx -g 1 -auto-alt-ref 0 -b:v 200k -c:a libopus vp8-key-opus.webm
F $V -frames:v 9 -c:v libvpx -g 30 -b:v 200k vp8-interframe.webm
F $V -frames:v 5 -c:v libvpx-vp9 -b:v 200k vp9.webm
F $V -frames:v 5 -c:v libx264 -pix_fmt yuv420p tiny.mp4
```

The webm muxer refuses PCM audio, so the PCM16 clip is Matroska
(`vp8-key-pcm.mkv`); sd.cpp reads both doc types.
