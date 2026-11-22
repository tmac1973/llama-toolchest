# Test audio

A 1-second 440 Hz sine at 44.1 kHz stereo, made with ffmpeg 8:

```bash
S="sine=frequency=440:sample_rate=44100:duration=1"
ffmpeg -f lavfi -i $S -ac 2 -c:a pcm_s16le tone.wav
ffmpeg -f lavfi -i $S -ac 2 -c:a libmp3lame -b:a 64k tone.mp3
ffmpeg -f lavfi -i $S -ac 2 -c:a flac tone.flac
ffmpeg -f lavfi -i $S -ac 2 -c:a libvorbis -q:a 0 tone.ogg
```

`broken.mp3` is the ID3 tag of `tone.mp3` followed by 4 KB of zero bytes:
it looks like an MP3 but has no audio frames.
`tone.ogg` is a format the package refuses.
