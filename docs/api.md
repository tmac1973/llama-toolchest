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

The fields are llama-server's. Which ones a model uses depends on the model:

| Field | What it does | Qwen3-TTS |
| --- | --- | --- |
| `seed` | `-1` (the default) is random. Another seed gives another take. The same seed gives the same audio on the CPU, but on a GPU it can give a slightly different take. | Yes |
| `temperature`, `top_k`, `top_p` | Sampling. Leave them out for the server's and the model's own values. | Yes |
| `speaker_ref`, `speaker_ref_b64` | A recording of the voice to copy. | Yes |
| `speaker_ref_text` | What is said in that recording. The copy is closer. Needs llama.cpp b12491 or newer, a recording of 30 s or less, and a projector with the reference encoder. | Yes |
| `speed` | Pace as a factor of the model's own (`1.0`). Needs b12238 or newer. | No: any value other than `1.0` gets a `400` |
| `lang`, `voice`, `response_format`, `stream` | Language, a voice of the model's own, `wav`/`pcm`/`pcm16`, and audio sent as it is made. | `lang` yes, `voice` no |

A model that cannot change its pace answers `400` for `speed`, so a client that needs another pace has to time-stretch the audio itself. The API reference (`/docs/api`) has every field. llama-server also answers the OpenAI-style `POST /v1/audio/speech` with the same models.

## Image, video and music API { #generation-api }

Instances show in `/v1/models` with `meta.kind` `image`, `video`, `music` or `upscale`. Use the instance or set ID as `model`. `meta.api` lists the routes of each instance, `meta.video` the limits of a video instance, and `meta.tasks` the tasks of a music instance. `meta.max_edit_images` gives the most reference images of an edit, and `meta.scale` the factor of an upscaler. `meta.measuring` is true while an instance is being measured. It refuses requests until the measurement ends. The API reference (`/docs/api`) has every field and error.

| Kind | Wait for the result | Job (answers at once with an ID) | Download |
| --- | --- | --- | --- |
| Image | `POST /v1/images/generations`, `/v1/images/edits` | `POST /v1/images/jobs` | `…/jobs/{id}/image/0` |
| Upscale | `POST /v1/images/upscale` | – | In the answer |
| Video | `POST /v1/videos/generations` | `POST /v1/videos/jobs` | `…/jobs/{id}/video` |
| Music | – | `POST /v1/music/jobs`, or the ACE-Step routes `/release_task`, `/query_result`, `/v1/audio` | `…/jobs/{id}/audio` |

- **Jobs:** `GET` a job until `status` is `succeeded`, `failed` or `cancelled`. `phase` shows progress (`starting`, `queued`, `generating` or `composing`/`synthesizing`, `saving`) and `queue_position` the jobs ahead. `GET /v1/{images,videos,music}/jobs?status=…` lists jobs.
- `DELETE` cancels an unfinished job or deletes a finished one. A running image cannot stop; a running video can. Results stay 24 hours, up to 200 image or music jobs and 50 video jobs. Jobs survive a restart, but unfinished ones show `failed`.
- **Image:** a job takes the JSON of `/v1/images/generations`, or the multipart form of `/v1/images/edits` for an edit. Edits need a set with the `edit` capability.
- **Upscale:** JSON with `image` (base64) and, if you like, `repeats` (1–4). It needs an upscaler set, such as Real-ESRGAN x4plus: each run makes the image 4 times wider and higher, up to 8192 pixels. The result has no transparency.
- **Video:** JSON or multipart with `prompt`, `size`, `frames` or `seconds`, `fps`, `start_image`, `end_image`, `output_format` (`webm`, with sound for LTX-2.3, or `webp`), `extra`. Requests above the measured size are refused (`over_measured_size`).
- **Music:** `caption`, `lyrics`, `duration`. To work on an existing song, set `task_type` (`cover`, `repaint`, `lego`, `extract`, `complete`) and post it as multipart with an `audio` part. `lego`, `extract` and `complete` need a base model.

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

**Transparent backgrounds with Ming-Image** need both: a prompt that starts with `RGBA, 4-channel, transparent background.`, and a fully transparent start image of the output size at strength 0.9, in an `<sd_cpp_extra_args>` block (84–89% of pixels came back clear):

```json
{"model": "ming-image", "size": "1024x1024", "response_format": "b64_json",
 "prompt": "<sd_cpp_extra_args>{\"init_image\": \"iVBORw0KGgo…\", \"strength\": 0.9}</sd_cpp_extra_args>RGBA, 4-channel, transparent background. a red fox sticker"}
```

*Open web UI →* on the Server page opens the own web page of each backend and each running fork instance (port 8081 and up). In a container, toolchest publishes those ports only if you chose that in `setup.sh`.
