# Measured reservations

These are the reservations that Measure gave for the catalog sets, on two test cards: a 32 GB Radeon AI PRO R9700 (ROCm) and a 16 GB RTX A4000 (CUDA). Measure on your own card: the figures change with the card, the driver and the backend version. See [Generation reservations](../gpu/reservations.md) for how Measure works.

October 2026, sd.cpp `master-1015`, acestep.cpp `de858d8`. Video figures are for the Short size unless noted. Times are on the R9700. The R9700 figures are from 2026-10-29, with text models on the CPU. So are the A4000 figures for FLUX.2 klein, TI2V 5B and LTX-2.3. The others are older, with text models on the GPU.

## The table { #table }

| Model set | Makes | R9700 32 GB | A4000 16 GB | Size measured | Time per job | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Ming-Image 0.1 Design | Images, transparent backgrounds | 10.5 GiB | 10.5 GiB |  |  | Text model runs on the CPU: half the memory, same speed |
| Qwen-Image 2.1 Turbo | Images | 6.5 GiB | – |  |  |  |
| Qwen-Image 2.1 Turbo Edit | Images, edits | 13.5 GiB | – |  | ~34 s image, ~88 s edit at 1024² | Edits need about twice the memory of images. Its picture reader runs on the CPU with the text model |
| Z-Image Turbo | Images | 8.5 GiB | – |  |  |  |
| FLUX.2 klein 4B | Images, edits | 8.5 GiB | 7.5 GiB |  | 5.1 s at 1024² (A4000) | Both figures include an edit |
| FLUX.3 Image Klein | Images, edits | 9.5 GiB | – |  |  | Edits need 2–3 GiB more than images |
| Chroma1-HD | Images | 12.0 GiB | – |  |  |  |
| Qwen-Image-Edit 2511 | Images, edits | 16.5 GiB | – |  |  | Edits need 2–3 GiB more than images. Its picture reader runs on the CPU with the text model: about 17 s more per edit |
| Wan2.2 TI2V 5B | Video | 20.0 GiB | 13.8 GiB | Short: 33 frames, 832×480 | ~70 s; ~14 min Medium inside 14 GiB; ~110 s Short (A4000) | Medium and Full need more than the R9700 has with the whole room, but fit when measured inside a 14 GiB budget. A4000: Medium crashed in stable-diffusion.cpp's CUDA code while decoding in smaller pieces |
| Wan2.2 I2V A14B | Video from a start (and end) image | 28.0 GiB | Fails | Short: 33 frames, 832×480 | ~27 min | Medium: the decode does not fit the R9700. Full not measured. A4000: tiled decode crashes |
| Wan2.2 I2V A14B Lightning | Video from a start (and end) image | 28.3 GiB | Fails | Short: 33 frames, 832×480 | ~3 min 20 s | A little less detail than A14B. Medium needs 30.5 GiB: too much for the R9700 |
| LTX-2.3 distilled 1.1 | Video with sound | 27.7 GiB Full | 13.8 GiB Full | Short: 33 frames, 768×512; Full: 5 s, 1280×720 | ~35 s Short; ~4 min 20 s Full (~11 min with start and end images) | A4000: Medium and Full fit, with most of the model in system RAM; stable-diffusion.cpp's CUDA code still crashes there now and then |
| HunyuanVideo 1.5 I2V 480p step-distilled | Video from a start image | 27.5 GiB | Crashes | Up to Full: 5 s, 848×480 | ~2 min Short | The fastest HunyuanVideo set: 8 steps. A4000: stable-diffusion.cpp crashed at the start of every video (CUDA), 3 times in 3. |
| HunyuanVideo 1.5 480p, CFG-distilled (I2V, T2V) | Video from text or a start image | 27.5 GiB | Crashes (expected) | Up to Full: 5 s, 848×480 | ~8 min Short | 50 steps, one pass each. A4000: not tried. It shares its code with the step-distilled set, which crashed there at the start of every video (CUDA). |
| HunyuanVideo 1.5 480p, base (I2V, T2V) | Video from text or a start image | 27.5 GiB | Crashes (expected) | Up to Full: 5 s, 848×480 | ~16 min Short | 50 steps, two passes each. A4000: not tried. It shares its code with the step-distilled set, which crashed there at the start of every video (CUDA). |
| HunyuanVideo 1.5 720p (I2V CFG-distilled, I2V, T2V) | Video from text or a start image | 13–14.5 GiB | Crashes (expected) | Short only: 33 frames, 1280×720 | ~37 min Short (I2V CFG-distilled), ~70 min (T2V) | The 720p decode needs 31.4 GB, so it is done in smaller pieces. 65 frames ran out of memory; 121 frames crashed in stable-diffusion.cpp. A4000: not tried. It shares its code with the step-distilled set, which crashed there at the start of every video (CUDA). |
| ACE-Step 1.5 XL SFT | Music | 8.0 GiB | – |  |  |  |
| ACE-Step 1.5 turbo | Music | – | 7.5 GiB |  | ~13 s per 60 s song (A4000) |  |
| Real-ESRGAN x4plus | Upscales images 4× | 1.0 GiB | – | 1024² test image | ~13 s for 768² to 3072² | No prompt. The result has no transparency |

## Notes { #notes }

- **Text encoders on the CPU.** All catalog image and video sets except SDXL run their text encoder on the CPU (`--backend te=cpu`). On the R9700 this saved 4–5.5 GiB per image set at no real cost. It saved 6 GiB on TI2V 5B and 3 GiB on LTX-2.3 at Full, enough to make Full fit; each video takes 13–16 s longer. Wan2.2 A14B saves only about 1 GiB, because sd.cpp already removes its encoder from VRAM before sampling. An LLM partly on the CPU and an image job do not slow each other more than before: the encoder runs for only a second or two. A custom set can do the same with `--backend te=cpu` in its launch arguments. Press *Details* on a set to see why it uses each of its settings.
- **Which HunyuanVideo 1.5 set.** The I2V 480p step-distilled set is the fastest: about 2 minutes for a short video. A 480p CFG-distilled set gives 50-step quality in half the time of the base set. A 720p set gives more detail when a short clip is enough. I2V sets need a start image; T2V sets need only text. Their licence, the Tencent Hunyuan Community License, is not valid in the EU, the UK or South Korea. They do not work on NVIDIA cards yet: stable-diffusion.cpp crashed ("CONCAT failed") at the start of every video on an RTX A4000.
- **System RAM.** An instance row on the Models page shows the system RAM that the backend used in its measurement: the text encoder, and weights that sd.cpp kept in RAM. Other programs need that RAM too.
- **sd-server and ace-server.** sd-server stays within its reservation. The ace-server figure is for planning only. On a smaller card, sd-server keeps part of a set in system RAM, so measure on the card you will use.

