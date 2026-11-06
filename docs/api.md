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

*Open web UI →* on the Server page opens the own web page of each backend (port 8081 and up). In a container, toolchest publishes those ports only if you chose that in `setup.sh`.
