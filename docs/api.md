# Using the API

The server speaks the OpenAI `/v1/*` API. Point any OpenAI-compatible client at it. The full API reference is in the app, at `/docs/api` on your own install. It lists every route with a form to try it, and `/openapi.json` has the same for code generators.

```bash
curl $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth-Qwen3.5-4B.Q4_K_M",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

## Default model name { #default-model }

Set a client's model to `default` and it works with whichever chat model you use, so you don't have to edit each client when you switch models. Real model names and aliases still work as before. On each chat request, `default` becomes one real model, picked in this order:

1. The model or fork instance you pinned.
2. The loaded chat model, or running fork, that answered a chat request most recently.
3. The chat model you used last before the server restarted. It loads with the usual room checks.

If none of these exists, the request gets a `503` with the code `no_default_model`. A pinned model that is not loaded loads as it would under its own name. A pinned fork that is stopped answers `503`, as a fork does. When a pinned model is deleted or disabled, or its fork instance is removed, the pin is cleared.

The name applies to `/chat/completions`, `/completions`, `/messages` and `/responses`, with or without `/v1`. A chat request with no `model` field counts as `default`. On any other route, `default` gets a `400` with the code `default_model_chat_only`, because embedding, reranking and generation models keep their own names. The response's `model` field names the model that actually answered.

`/v1/models` lists the name first, so a client's model picker shows it. Its `meta` says that it's virtual and which model it goes to right now, and why: `pinned`, `loaded` (the loaded model used last) or `last used` (the next request loads it). The entry is left out while the name is off or has nowhere to go.

```json
{"id": "default", "object": "model", "created": 0, "owned_by": "llama-toolchest",
 "meta": {"kind": "text", "virtual": true, "target": "unsloth-gemma-4-12b-it.UD_Q6_K_XL", "target_reason": "loaded"}}
```

If a model, alias or fork instance already has the name `default`, that one wins and the virtual name stays off until you rename it. Set the name and the pin in [Settings → Default model](guides/settings.md#default-model). They are kept in the config file as:

```yaml
default_model:
  name: default    # what clients send; empty means "default"
  pin: ""          # a model ID, or fork:<instance ID>; empty means no pin
  off: false       # true turns the name off
```

## Speech API { #speech-api }

`POST /tts` goes to llama-server's own `/tts` as it is, behind the same API key as `/v1/*`. If the model is not loaded, toolchest loads it first. The answer is a WAV file. Send JSON, or a multipart form with the voice to copy as a `speaker_ref` file. Get a model from [Get speech models](guides/download-models.md#speech-models).

```bash
curl $BASE/tts -o line.wav \
  -F model=moonbox-dev-Qwen3-TTS-12Hz-1.7B-Base.Q8_0 -F input="The ship left at dawn." -F lang=en \
  -F speaker_ref=@voice.wav -F speaker_ref_text="What is said in voice.wav." -F seed=7
```

With a model that has voices of its own, name one instead of sending a recording:

```bash
curl $BASE/tts -o line.wav -H 'Content-Type: application/json' \
  -d '{"model": "ggml-org-Kokoro-82M.F32", "input": "The ship left at dawn.", "voice": "am_michael", "speed": 1.1}'
```

The fields are llama-server's. Which ones a model uses depends on the model:

| Field | What it does | Models that use it |
| --- | --- | --- |
| `seed` | `-1` (the default) is random. Another seed gives another take. The same seed gives the same audio on the CPU, but on a GPU it can give a slightly different take. | All |
| `temperature`, `top_k`, `top_p` | Sampling. Leave them out for the server's and the model's own values. | Qwen3-TTS. ZONOS3 and VoxCPM2 take only `temperature`. Kokoro and S2 Pro use their own values. |
| `voice` | One of the model's own voices. llama-server's `GET /props` lists them. Not with `speaker_ref`. | Kokoro (54), ZONOS3 (27), Chatterbox (`default`) |
| `speaker_ref`, `speaker_ref_b64` | A recording of the voice to copy. | Qwen3-TTS, Chatterbox, ZONOS3, VoxCPM2, S2 Pro. Kokoro answers `400`. |
| `speaker_ref_text` | What is said in that recording. The copy is closer. Qwen3-TTS needs a recording of 30 s or less, llama.cpp b12491 or newer, and a projector with the reference encoder. S2 Pro uses the first 30 s. | Qwen3-TTS, S2 Pro |
| `speed` | Pace as a factor of the model's own (`1.0`). Needs b12238 or newer. | Kokoro, ZONOS3. The others answer `400` for any value other than `1.0`. |
| `loudness` | Target loudness in LUFS, below `0`. | ZONOS3 |
| `flow_steps`, `cfg_scale` | Steps of the audio generator (2 to 32, default 10), and how closely it follows the text (default 2.0). Fewer steps are faster. | VoxCPM2 |
| `raw_prompt` | Give `input` to the model as it is, without its own text preparation. With Kokoro, use it to send phonemes. | Kokoro, ZONOS3 |
| `lang` | Language of the text. | Qwen3-TTS, Kokoro (`en-us`, `en-gb`), ZONOS3 (numbers and times only). The others read the language from the text. |
| `response_format`, `stream` | `wav`/`pcm`/`pcm16`, and audio sent as it is made. | All |

A model that cannot change its pace answers `400` for `speed`, so a client that needs another pace has to time-stretch the audio itself. Some models read tags in the text: VoxCPM2 a voice description in brackets at the start, S2 Pro tags such as `[laughing]` and speaker tags such as `<|speaker:0|>`. The API reference (`/docs/api`) has every field. llama-server also answers the OpenAI-style `POST /v1/audio/speech` with the same models.

## Image, video and music API { #generation-api }

Instances show in `/v1/models` with `meta.kind` `image`, `video`, `music`, `upscale` or `vupscale`. Use the instance or set ID as `model`. `meta.api` lists the routes of each instance, `meta.video` the limits of a video instance, and `meta.tasks` the tasks of a music instance. `meta.max_edit_images` gives the most reference images of an edit, and `meta.scale` the factor of an upscaler. `meta.measuring` is true while an instance is being measured. It refuses requests until the measurement ends. The API reference (`/docs/api`) has every field and error.

| Kind | Wait for the result | Job (answers at once with an ID) | Download |
| --- | --- | --- | --- |
| Image | `POST /v1/images/generations`, `/v1/images/edits` | `POST /v1/images/jobs` | `…/jobs/{id}/image/0` |
| Upscale | `POST /v1/images/upscale` | – | In the answer |
| Video | `POST /v1/videos/generations` | `POST /v1/videos/jobs` | `…/jobs/{id}/video` |
| Video upscale | – | `POST /v1/videos/upscale/jobs` | `…/jobs/{id}/video` |
| Music | – | `POST /v1/music/jobs`, or the ACE-Step routes `/release_task`, `/query_result`, `/v1/audio` | `…/jobs/{id}/audio` |

- **Jobs:** `GET` a job until `status` is `succeeded`, `failed` or `cancelled`. `phase` shows progress (`starting`, `queued`, `generating` or `composing`/`synthesizing`, `saving`) and `queue_position` the jobs ahead. `GET /v1/{images,videos,music}/jobs?status=…` lists jobs.
- `DELETE` cancels an unfinished job or deletes a finished one. A running image cannot stop; a running video can, and a running video upscale stops its instance. Results stay 24 hours, up to 200 image or music jobs and 50 video or video upscale jobs. Jobs survive a restart, but unfinished ones show `failed`.
- **Image:** a job takes the JSON of `/v1/images/generations`, or the multipart form of `/v1/images/edits` for an edit. Edits need a set with the `edit` capability.
- **Upscale:** JSON with `image` (base64) and, if you like, `repeats` (1–4), `tile_size` and `output_format`. It needs an upscaler set, such as Real-ESRGAN x4plus: each run makes the image 4 times wider and higher, up to 8192 pixels. stable-diffusion.cpp master-1119 removed sd-server's own upscale route, so on builds with DGL (master-1116 and newer) toolchest runs the upscale as a DGL script. The answer has the same form, but the image must be PNG, JPEG or GIF, and the result is PNG or JPEG (no WebP). The upscaler sees only the colors; when the image has transparency, a PNG result gets it back, scaled to the new size. A JPEG result, and the result on older builds, has no transparency.
- **Video:** JSON or multipart with `prompt`, `size`, `frames` or `seconds`, `fps`, `start_image`, `end_image`, `audio`, `output_format` (`webm`, with sound for LTX-2.3 and LTX-2.5, or `webp`), `extra`. Requests above the measured size are refused (`over_measured_size`).
- **Video upscale:** a multipart form with the clip as the `video` file (up to 512 MiB), or JSON with `video` as base64 or a data URL. `model` is a video upscaler instance or set (`meta.kind` `vupscale`), and `output_format` is `webm` (default, no sound) or `avi` (keeps the sound). `seed` and `window_size` are optional. The clip must be AVI (RGB or MJPEG), WebM with VP8 keyframes, or WebP; MP4 is refused. It is checked before it reaches the GPU, against stable-diffusion.cpp's limits and the size the instance was measured with, and the error says what to change. `meta.vupscale` gives the scale, the window sizes and the measured size. Cancelling a running upscale cancels it in stable-diffusion.cpp (master-1132 and newer); on older builds, or when the job has not stopped after 30 seconds, toolchest stops the instance instead. See [Video upscaling](guides/generation.md#video-upscaling).
- **Speech-to-video:** a set with the `audio` capability (Wan2.2 S2V) needs a `start_image` (the face) and an `audio` track: a WAV, MP3 or FLAC file, as a multipart file or base64. With no `frames` or `seconds`, the video is as long as the audio. The model makes only 16 fps (`meta.video.fixed_fps`), so a different `fps` is replaced. sd-server builds from stable-diffusion.cpp 6cd0ef1 (2026-11-22) put the audio in the WebM. Older builds make a silent video, so add the track yourself, for example with `ffmpeg -i video.webm -i speech.wav -c:v copy -c:a libopus -shortest out.webm`.
- **Music:** `caption`, `lyrics`, `duration`. To work on an existing song, set `task_type` (`cover`, `repaint`, `lego`, `extract`, `complete`) and post it as multipart with an `audio` part. `lego`, `extract` and `complete` need a base model.

Upscale a clip two times with RelayVSR, keep its sound, and download the result:

```bash
curl $BASE/v1/videos/upscale/jobs -F model=relayvsr-x2 -F video=@clip.webm -F output_format=avi
# {"id": "7c2e…", "status": "queued", "phase": "queued", "queue_position": 0}
curl $BASE/v1/videos/upscale/jobs/7c2e…
curl $BASE/v1/videos/upscale/jobs/7c2e…/video -o clip-2x.avi
```

```bash
curl $BASE/v1/images/jobs -H "Content-Type: application/json" \
  -d '{"model": "ming-image", "prompt": "a red fox", "size": "1024x1024"}'
# {"id": "3f9c…", "status": "queued", "phase": "queued", "queue_position": 2}
curl $BASE/v1/images/jobs/3f9c…
curl -o fox.png $BASE/v1/images/jobs/3f9c…/image/0
curl $BASE/v1/images/jobs -F model=qwen-image-edit-2511 -F prompt="make the fox blue" -F "image[]=@fox.png"
jq -n --rawfile img <(base64 -w0 fox.png) '{model: "real-esrgan-x4plus", image: $img}' |
  curl $BASE/v1/images/upscale -H "Content-Type: application/json" -d @- | jq -r '.images[0].b64_json' | base64 -d > fox-4x.png

curl $BASE/v1/videos/generations -F prompt="the waves start to roll" -F start_image=@photo.png
```

**Transparent backgrounds with Ming-Image**{ #ming-transparent } need both: a prompt that starts with `RGBA, 4-channel, transparent background.`, and a fully transparent start image of the output size at strength 0.9, in an `<sd_cpp_extra_args>` block. With 0.1 (`ming-image`), 84–89% of pixels came back clear; with 0.2 (`ming-image-0.2`), about 94%, with cleaner edges:

```json
{"model": "ming-image", "size": "1024x1024", "response_format": "b64_json",
 "prompt": "<sd_cpp_extra_args>{\"init_image\": \"iVBORw0KGgo…\", \"strength\": 0.9}</sd_cpp_extra_args>RGBA, 4-channel, transparent background. a red fox sticker"}
```

*Open web UI →* on the Server page opens the own web page of each backend and each running fork instance (port 8081 and up). In a container, toolchest publishes those ports only if you chose that in `setup.sh`.
