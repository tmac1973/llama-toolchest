package api

// Music generation API. Two route sets feed the same jobs and the same
// per-GPU queue:
//
//   - /v1/music/jobs: llama-toolchest's own REST API, for Haruspex and any
//     other client (there is no standard music API).
//   - /release_task, /query_result and /v1/audio: the official ACE-Step 1.5
//     server's API, so tools written for that server can point here. The
//     shapes follow https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md
//     on the main branch as of 2026-10-24.
//
// Both wrap acestep.cpp's own job API (/lm, /synth, /job), which clients
// never see.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// Music job statuses.
const (
	musicQueued    = jobQueued
	musicRunning   = jobRunning
	musicSucceeded = jobSucceeded
	musicFailed    = jobFailed
	musicCancelled = jobCancelled
)

// Music job phases: what a job is doing while it is queued or running.
// A finished job has none.
const (
	musicPhaseQueued = "queued"
	// musicPhaseStarting: with eviction on, the job's backend was stopped
	// and is being started, after making room for it (gen_evict.go).
	musicPhaseStarting     = "starting"
	musicPhaseComposing    = "composing"
	musicPhaseSynthesizing = "synthesizing"
	musicPhaseSaving       = "saving"
)

// musicRequest is a music job's request, whichever route it came from.
type musicRequest struct {
	Model          string  `json:"model,omitempty"`
	Caption        string  `json:"caption"`
	Lyrics         string  `json:"lyrics,omitempty"`
	Duration       float64 `json:"duration,omitempty"`
	BPM            int     `json:"bpm,omitempty"`
	KeyScale       string  `json:"keyscale,omitempty"`
	TimeSignature  string  `json:"timesignature,omitempty"`
	VocalLanguage  string  `json:"vocal_language,omitempty"`
	Seed           *int64  `json:"seed,omitempty"`
	InferenceSteps int     `json:"inference_steps,omitempty"`
	GuidanceScale  float64 `json:"guidance_scale,omitempty"`
	// OutputFormat is "mp3" (the default) or "wav".
	OutputFormat string `json:"output_format,omitempty"`
	// RewriteCaption lets ACE-Step's language model rewrite the caption
	// before composing (ace's use_cot_caption). Off by default: on
	// asmodean it turned a request for a male vocal into a female one.
	RewriteCaption bool `json:"rewrite_caption,omitempty"`

	// TaskType is one of musicTasks; empty means text2music. Every task but
	// text2music works on source audio.
	TaskType string `json:"task_type,omitempty"`
	// Track is the stem lego, extract and complete work on, one of
	// musicTracks; complete takes several joined by " | ".
	Track string `json:"track,omitempty"`
	// AudioCoverStrength (0–1) is the share of steps that hear the source
	// in a cover: 1 keeps its structure, about 0.2 is a loose restyle.
	AudioCoverStrength *float64 `json:"audio_cover_strength,omitempty"`
	// CoverNoiseStrength (0–1) starts a cover from the source rather than
	// from noise.
	CoverNoiseStrength *float64 `json:"cover_noise_strength,omitempty"`
	// RepaintingStart and RepaintingEnd bound the part repaint and lego
	// redo, in seconds; an end of -1 is the end of the source. A start
	// below 0 or an end past the source adds music there (extend).
	RepaintingStart *float64 `json:"repainting_start,omitempty"`
	RepaintingEnd   *float64 `json:"repainting_end,omitempty"`
	// SourceAudio and ReferenceAudio say whether the request came with a
	// source song (what the task works on) and a timbre reference. The
	// files are held only until the job reaches ace-server.
	SourceAudio    bool `json:"source_audio,omitempty"`
	ReferenceAudio bool `json:"reference_audio,omitempty"`
	source, ref    *aceFile
}

// Music tasks, as acestep.cpp names them (src/task-types.h).
const (
	taskText2Music = "text2music"
	taskCover      = "cover"
	taskCoverNoFSQ = "cover-nofsq"
	taskRepaint    = "repaint"
	taskLego       = "lego"
	taskExtract    = "extract"
	taskComplete   = "complete"
)

var musicTasks = []string{taskText2Music, taskCover, taskCoverNoFSQ, taskRepaint, taskLego, taskExtract, taskComplete}

// stemTasks need a base or SFT DiT: acestep.cpp warns that on turbo their
// output is incoherent.
var stemTasks = []string{taskLego, taskExtract, taskComplete}

// musicTracks are the stems ACE-Step was trained on.
var musicTracks = []string{"vocals", "backing_vocals", "drums", "bass", "guitar", "keyboard",
	"percussion", "strings", "synth", "fx", "brass", "woodwinds"}

// musicUploadLimit caps a multipart music request: about ten minutes of
// 48 kHz stereo WAV, which is ace-server's longest source (15000 latent
// frames).
const musicUploadLimit = 128 << 20

// aceTasks returns the tasks a music set can run. A turbo DiT, which
// acestep.cpp tells apart by its metadata, is named so in every ACE-Step
// GGUF we know of.
func aceTasks(set generation.ModelSet) []string {
	for _, c := range set.Components {
		if c.Slot == "dit" && strings.Contains(strings.ToLower(filepath.Base(firstNonEmpty(c.Path, c.File))), "turbo") {
			return slices.DeleteFunc(slices.Clone(musicTasks), func(t string) bool { return slices.Contains(stemTasks, t) })
		}
	}
	return musicTasks
}

func (q musicRequest) task() string {
	return firstNonEmpty(q.TaskType, taskText2Music)
}

// checkMusicRequest refuses a request the set can't run, with an error
// code and message; captionField is the caption's name on the route.
func checkMusicRequest(set generation.ModelSet, q musicRequest, captionField string) (code, msg string) {
	task := q.task()
	bad := func(m string) (string, string) { return "bad_request", m }
	if !slices.Contains(musicTasks, task) {
		return bad("task_type must be one of " + strings.Join(musicTasks, ", "))
	}
	if !slices.Contains(aceTasks(set), task) {
		return "task_not_supported", task + " needs a base or SFT model; " + set.ID + " is turbo"
	}
	if task != taskExtract && strings.TrimSpace(q.Caption) == "" {
		return bad(captionField + " is required: a short description of the music")
	}
	if task == taskText2Music && q.source != nil {
		return bad("source audio is only used by the other tasks; set task_type")
	}
	if task != taskText2Music && q.source == nil {
		return bad(task + " needs source audio: send a multipart request with it in the audio part")
	}
	if q.Track != "" {
		if !slices.Contains(stemTasks, task) {
			return bad("track is only used by lego, extract and complete")
		}
		names := []string{q.Track}
		if task == taskComplete {
			names = strings.Split(q.Track, " | ")
		}
		for _, n := range names {
			if !slices.Contains(musicTracks, n) {
				return bad(fmt.Sprintf("unknown track %q; tracks are %s", n, strings.Join(musicTracks, ", ")))
			}
		}
	}
	for name, v := range map[string]*float64{"audio_cover_strength": q.AudioCoverStrength, "cover_noise_strength": q.CoverNoiseStrength} {
		if v == nil {
			continue
		}
		if task != taskCover && task != taskCoverNoFSQ {
			return bad(name + " is only used by cover")
		}
		if *v < 0 || *v > 1 {
			return bad(name + " must be between 0 and 1")
		}
	}
	if q.RepaintingStart != nil || q.RepaintingEnd != nil {
		if task != taskRepaint && task != taskLego {
			return bad("repainting_start and repainting_end are only used by repaint and lego")
		}
		start, end := 0.0, -1.0
		if q.RepaintingStart != nil {
			start = *q.RepaintingStart
		}
		if q.RepaintingEnd != nil {
			end = *q.RepaintingEnd
		}
		if end != -1 && end <= start {
			return bad("repainting_end must be after repainting_start, or -1 for the end of the source")
		}
	}
	return "", ""
}

// aceRequest is the request for ace-server's /lm step, or for /synth when
// a task works on source audio and skips /lm. Zero values are left out so
// ace-server picks its own.
func (q musicRequest) aceRequest() map[string]any {
	req := map[string]any{"caption": q.Caption, "use_cot_caption": q.RewriteCaption}
	set := func(k string, v any, ok bool) {
		if ok {
			req[k] = v
		}
	}
	set("lyrics", q.Lyrics, q.Lyrics != "")
	set("duration", q.Duration, q.Duration > 0)
	set("bpm", q.BPM, q.BPM > 0)
	set("keyscale", q.KeyScale, q.KeyScale != "")
	set("timesignature", q.TimeSignature, q.TimeSignature != "")
	set("vocal_language", q.VocalLanguage, q.VocalLanguage != "")
	set("inference_steps", q.InferenceSteps, q.InferenceSteps > 0)
	set("guidance_scale", q.GuidanceScale, q.GuidanceScale > 0)
	if q.Seed != nil {
		req["seed"], req["lm_seed"] = *q.Seed, *q.Seed
	}
	if q.task() != taskText2Music {
		req["task_type"] = q.task()
		set("track", q.Track, q.Track != "")
		for k, v := range map[string]*float64{
			"audio_cover_strength": q.AudioCoverStrength, "cover_noise_strength": q.CoverNoiseStrength,
			"repainting_start": q.RepaintingStart, "repainting_end": q.RepaintingEnd,
		} {
			if v != nil {
				req[k] = *v
			}
		}
	}
	return req
}

// files returns the request's uploads as ace-server's /synth parts.
func (q musicRequest) files() []aceFile {
	var out []aceFile
	if q.source != nil {
		out = append(out, *q.source)
	}
	if q.ref != nil {
		out = append(out, *q.ref)
	}
	return out
}

func (q musicRequest) ext() string {
	if q.OutputFormat == "wav" {
		return "wav"
	}
	return "mp3"
}

// musicJob is one song request.
type musicJob struct {
	ID         string         `json:"id"`
	InstanceID string         `json:"model"`
	Status     string         `json:"status"`
	Phase      string         `json:"phase,omitempty"`
	Position   int            `json:"queue_position"`
	Request    musicRequest   `json:"request"`
	Result     map[string]any `json:"result,omitempty"`
	AudioURL   string         `json:"audio_url,omitempty"`
	Error      string         `json:"error,omitempty"`
	Submitted  time.Time      `json:"submitted"`
	Started    *time.Time     `json:"started,omitempty"`
	Finished   *time.Time     `json:"finished,omitempty"`

	// Not shown to clients.
	audioPath    string
	backendJobID string
	cancel       context.CancelFunc
	ticket       *generation.Ticket
	lmModel      string
	ditModel     string
}

func (j musicJob) state() jobState {
	return jobState{id: j.ID, status: j.Status, ticket: j.ticket, finished: j.Finished, files: []string{j.audioPath}}
}

func (j musicJob) withPosition(n int) musicJob {
	j.Position = n
	return j
}

func (s *Server) musicDir() string {
	return filepath.Join(s.cfg.DataDir, "gen-output", "music")
}

// submitMusic queues a song on the instance's GPU and runs it in the
// background. It fails at once when the GPU's queue is full.
func (s *Server) submitMusic(inst generation.Instance, req musicRequest) (*musicJob, error) {
	if req.OutputFormat != "" && req.OutputFormat != "mp3" && req.OutputFormat != "wav" {
		return nil, fmt.Errorf("output_format must be mp3 or wav")
	}
	ticket, err := s.genQueue.Submit(inst.GPU, inst.ID, generation.KindAce)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	job := &musicJob{
		ID: newGenJobID(), InstanceID: inst.ID, Status: musicQueued, Phase: musicPhaseQueued, Request: req,
		Submitted: ticket.Info().Submitted, cancel: cancel, ticket: ticket,
	}
	// A free GPU starts the job at once; say so in the first answer.
	if started := ticket.Info().Started; !started.IsZero() {
		job.Status, job.Phase, job.Started = musicRunning, req.firstPhase(), &started
		if !s.genMgr.IsRunning(inst.ID) {
			job.Phase = musicPhaseStarting
		}
	}
	if set, err := s.gen.ModelSet(inst.ModelSetID); err == nil {
		for _, c := range set.Components {
			name := strings.TrimSuffix(filepath.Base(firstNonEmpty(c.Path, c.File)), ".gguf")
			switch c.Slot {
			case "lm":
				job.lmModel = name
			case "dit":
				job.ditModel = name
			}
		}
	}
	s.genMusic.add(job)
	go s.runMusicJob(ctx, job.ID, inst, ticket)
	return job, nil
}

// firstPhase is a running job's first phase: a task on source audio has no
// composing step.
func (q musicRequest) firstPhase() string {
	if q.task() == taskText2Music {
		return musicPhaseComposing
	}
	return musicPhaseSynthesizing
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// runMusicJob waits for the GPU, then runs ace-server's /lm (lyrics and
// arrangement) and /synth (audio) steps and saves the audio. A task on
// source audio skips /lm, as the official server does: the source stands
// in for the arrangement /lm would write.
func (s *Server) runMusicJob(ctx context.Context, id string, inst generation.Instance, ticket *generation.Ticket) {
	finish := func(status, errMsg string) {
		now := time.Now()
		s.genMusic.update(id, func(j *musicJob) {
			j.Request.source, j.Request.ref = nil, nil
			if j.Status == musicCancelled {
				return
			}
			j.Status, j.Phase, j.Error, j.Finished = status, "", errMsg, &now
		})
		if status == musicFailed {
			slog.Warn("music job failed", "job", id, "instance", inst.ID, "error", errMsg)
		}
	}
	if err := ticket.Wait(ctx); err != nil {
		finish(musicCancelled, "")
		return
	}
	defer ticket.Done()
	if !s.genMgr.IsRunning(inst.ID) {
		now := time.Now()
		s.genMusic.update(id, func(j *musicJob) {
			if j.Status == musicCancelled {
				return
			}
			if j.Started == nil {
				j.Started = &now
			}
			j.Status, j.Phase = musicRunning, musicPhaseStarting
		})
	}
	if err := s.genReady(ctx, inst); err != nil {
		finish(musicFailed, "starting the backend: "+err.Error())
		return
	}

	view, _ := s.genMusic.view(id)
	now := time.Now()
	cancelled := false
	s.genMusic.update(id, func(j *musicJob) {
		if j.Status == musicCancelled {
			cancelled = true
			return
		}
		if j.Started == nil {
			j.Started = &now
		}
		j.Status, j.Phase = musicRunning, j.Request.firstPhase()
	})
	if cancelled {
		return
	}
	setPhase := func(phase string) {
		s.genMusic.update(id, func(j *musicJob) {
			if j.Finished == nil {
				j.Phase = phase
			}
		})
	}
	setBackendJob := func(bid string) { s.genMusic.update(id, func(j *musicJob) { j.backendJobID = bid }) }
	base := fmt.Sprintf("http://127.0.0.1:%d", inst.Port)

	fail := func(step string, err error) {
		switch {
		case errors.Is(err, context.Canceled):
			finish(musicCancelled, "")
		case !s.genMgr.IsRunning(inst.ID):
			finish(musicFailed, "backend stopped")
		default:
			finish(musicFailed, step+": "+err.Error())
		}
	}

	synthReq := view.Request.aceRequest()
	if view.Request.task() == taskText2Music {
		lmOut, _, err := runAceJob(ctx, base, "/lm", synthReq, setBackendJob)
		if err != nil {
			fail("composing", err)
			return
		}
		var enriched any
		if err := json.Unmarshal(lmOut, &enriched); err != nil {
			fail("composing", fmt.Errorf("reading the result: %w", err))
			return
		}
		if list, ok := enriched.([]any); ok && len(list) > 0 {
			enriched = list[0]
		}
		var ok bool
		if synthReq, ok = enriched.(map[string]any); !ok {
			fail("composing", errors.New("unexpected result"))
			return
		}
	}
	result := map[string]any{}
	for _, k := range []string{"caption", "lyrics", "bpm", "duration", "keyscale", "timesignature", "vocal_language", "seed"} {
		if v, ok := synthReq[k]; ok {
			result[k] = v
		}
	}
	s.genMusic.update(id, func(j *musicJob) { j.Result = result })
	setPhase(musicPhaseSynthesizing)

	synthReq["output_format"] = map[string]string{"mp3": "mp3", "wav": "wav16"}[view.Request.ext()]
	out, ctype, err := runAceJob(ctx, base, "/synth", synthReq, setBackendJob, view.Request.files()...)
	// ace-server has the uploads now, or the job is over.
	s.genMusic.update(id, func(j *musicJob) { j.Request.source, j.Request.ref = nil, nil })
	if err != nil {
		fail("synthesizing", err)
		return
	}
	audio, _, err := audioPart(ctype, out)
	if err != nil {
		fail("synthesizing", err)
		return
	}
	setPhase(musicPhaseSaving)
	if err := os.MkdirAll(s.musicDir(), 0o755); err != nil {
		fail("saving", err)
		return
	}
	file := filepath.Join(s.musicDir(), id+"."+view.Request.ext())
	if err := os.WriteFile(file, audio, 0o644); err != nil {
		fail("saving", err)
		return
	}
	s.genMusic.update(id, func(j *musicJob) { j.audioPath = file })
	finish(musicSucceeded, "")
}

// cancelMusic stops a job: a queued one leaves the queue, a running one is
// cancelled in ace-server.
func (s *Server) cancelMusic(id string) bool {
	var cancel context.CancelFunc
	found := false
	s.genMusic.update(id, func(j *musicJob) {
		found = true
		if j.Finished != nil {
			return
		}
		now := time.Now()
		cancel, j.Status, j.Phase, j.Finished = j.cancel, musicCancelled, "", &now
	})
	if cancel != nil {
		cancel()
	}
	return found
}

// musicResponse is a job as the /v1/music/jobs routes return it.
func musicResponse(j musicJob) musicJob {
	if j.Status == musicSucceeded {
		j.AudioURL = "/v1/music/jobs/" + j.ID + "/audio"
	}
	return j
}

// pickMusicInstance is pickInstance for music, as /v1/music/jobs uses it.
func (s *Server) pickMusicInstance(model string) (generation.Instance, *pickError) {
	return s.pickInstance(model, generation.KindAce)
}

func (s *Server) musicEnabled() bool {
	return s.cfg.GenerationEnabled && s.gen != nil
}

// checkMusic is checkMusicRequest for the instance's set.
func (s *Server) checkMusic(inst generation.Instance, req musicRequest, captionField string) (code, msg string) {
	set, err := s.gen.ModelSet(inst.ModelSetID)
	if err != nil {
		return "model_not_found", err.Error()
	}
	return checkMusicRequest(set, req, captionField)
}

// readMusicRequest reads a /v1/music/jobs body: JSON, or multipart with
// the JSON in a "request" part and the source and reference songs in
// "audio" and "ref_audio" parts, the names ace-server's /synth uses.
func readMusicRequest(w http.ResponseWriter, r *http.Request) (musicRequest, error) {
	var req musicRequest
	if !strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/") {
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
			return req, errors.New("the body is not valid JSON: " + err.Error())
		}
		req.SourceAudio, req.ReferenceAudio = false, false
		return req, nil
	}
	form, err := readMusicForm(w, r)
	if err != nil {
		return req, err
	}
	defer form.RemoveAll()
	raw := firstValue(form, "request")
	if raw == "" {
		if fh := firstFile(form, "request"); fh != nil {
			b, err := readFormFile(fh)
			if err != nil {
				return req, err
			}
			raw = string(b)
		}
	}
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		return req, errors.New("the request part is not valid JSON: " + err.Error())
	}
	req.source, req.ref, err = musicUploads(form, []string{"audio"}, []string{"ref_audio"})
	req.SourceAudio, req.ReferenceAudio = req.source != nil, req.ref != nil
	return req, err
}

func readMusicForm(w http.ResponseWriter, r *http.Request) (*multipart.Form, error) {
	r.Body = http.MaxBytesReader(w, r.Body, musicUploadLimit)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		return nil, fmt.Errorf("reading the multipart body (at most %d MB): %w", musicUploadLimit>>20, err)
	}
	return r.MultipartForm, nil
}

// musicUploads reads the source and reference songs from the first of
// their field names a form has.
func musicUploads(form *multipart.Form, srcFields, refFields []string) (src, ref *aceFile, err error) {
	read := func(fields []string, as string) (*aceFile, error) {
		for _, f := range fields {
			if fh := firstFile(form, f); fh != nil {
				b, err := readFormFile(fh)
				if err != nil {
					return nil, err
				}
				return &aceFile{field: as, name: fh.Filename, data: b}, nil
			}
		}
		return nil, nil
	}
	if src, err = read(srcFields, "audio"); err != nil {
		return nil, nil, err
	}
	ref, err = read(refFields, "ref_audio")
	return src, ref, err
}

func firstValue(form *multipart.Form, field string) string {
	if v := form.Value[field]; len(v) > 0 {
		return v[0]
	}
	return ""
}

func firstFile(form *multipart.Form, field string) *multipart.FileHeader {
	if f := form.File[field]; len(f) > 0 {
		return f[0]
	}
	return nil
}

func readFormFile(fh *multipart.FileHeader) ([]byte, error) {
	f, err := fh.Open()
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return io.ReadAll(f)
}

// --- /v1/music/jobs ---------------------------------------------------------

func (s *Server) handleMusicCreate(w http.ResponseWriter, r *http.Request) {
	if !s.musicEnabled() {
		openAIError(w, http.StatusNotFound, "invalid_request_error", "not_found", "music generation is not enabled on this server")
		return
	}
	req, err := readMusicRequest(w, r)
	if err != nil {
		openAIError(w, http.StatusBadRequest, "invalid_request_error", "bad_request", err.Error())
		return
	}
	inst, pe := s.pickMusicInstance(req.Model)
	if pe != nil {
		errType := "invalid_request_error"
		if pe.status >= 500 {
			errType = "server_error"
		}
		openAIError(w, pe.status, errType, pe.code, pe.msg)
		return
	}
	if code, msg := s.checkMusic(inst, req, "caption"); code != "" {
		openAIError(w, http.StatusBadRequest, "invalid_request_error", code, msg)
		return
	}
	job, err := s.submitMusic(inst, req)
	if errors.Is(err, generation.ErrQueueFull) {
		w.Header().Set("Retry-After", "30")
		openAIError(w, http.StatusTooManyRequests, "server_error", "queue_full", err.Error())
		return
	}
	if err != nil {
		openAIError(w, http.StatusBadRequest, "invalid_request_error", "bad_request", err.Error())
		return
	}
	view, _ := s.genMusic.view(job.ID)
	respondJSONStatus(w, http.StatusAccepted, map[string]any{"id": view.ID, "status": view.Status, "phase": view.Phase, "queue_position": view.Position})
}

func (s *Server) musicJobOr404(w http.ResponseWriter, id string) (musicJob, bool) {
	if !s.musicEnabled() {
		openAIError(w, http.StatusNotFound, "invalid_request_error", "not_found", "music generation is not enabled on this server")
		return musicJob{}, false
	}
	j, ok := s.genMusic.view(id)
	if !ok {
		openAIError(w, http.StatusNotFound, "invalid_request_error", "job_not_found", "no music job "+id)
	}
	return j, ok
}

func (s *Server) handleMusicGet(w http.ResponseWriter, r *http.Request) {
	if j, ok := s.musicJobOr404(w, chi.URLParam(r, "id")); ok {
		respondJSON(w, musicResponse(j))
	}
}

func (s *Server) handleMusicAudio(w http.ResponseWriter, r *http.Request) {
	j, ok := s.musicJobOr404(w, chi.URLParam(r, "id"))
	if !ok {
		return
	}
	if j.Status != musicSucceeded {
		openAIError(w, http.StatusConflict, "invalid_request_error", "not_ready", "the job is "+j.Status)
		return
	}
	serveMusicFile(w, r, j.audioPath)
}

func serveMusicFile(w http.ResponseWriter, r *http.Request, file string) {
	ctype := "audio/mpeg"
	if strings.HasSuffix(file, ".wav") {
		ctype = "audio/wav"
	}
	w.Header().Set("Content-Type", ctype)
	http.ServeFile(w, r, file)
}

func (s *Server) handleMusicDelete(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if _, ok := s.musicJobOr404(w, id); !ok {
		return
	}
	s.cancelMusic(id)
	w.WriteHeader(http.StatusNoContent)
}

// --- ACE-Step compatible routes ---------------------------------------------

// aceStepOK writes the official ACE-Step server's success wrapper.
func aceStepOK(w http.ResponseWriter, payload any) {
	respondJSON(w, map[string]any{"data": payload, "code": 200, "error": nil, "timestamp": time.Now().UnixMilli(), "extra": nil})
}

// aceStepError writes the official ACE-Step server's error shape.
func aceStepError(w http.ResponseWriter, status int, detail string) {
	respondJSONStatus(w, status, map[string]any{"detail": detail})
}

// pickAceStepInstance applies the compatibility routes' model rule: official
// clients send their own model names (e.g. acestep-v15-turbo), so a name
// that matches no instance or set falls back to the only eligible music
// instance.
func (s *Server) pickAceStepInstance(model string) (generation.Instance, int, string) {
	inst, pe := s.pickMusicInstance(model)
	if pe == nil {
		return inst, 0, ""
	}
	if pe.code == "model_not_found" && model != "" {
		inst, pe = s.pickMusicInstance("")
		if pe == nil {
			return inst, 0, ""
		}
	}
	if pe.status == http.StatusServiceUnavailable {
		return inst, pe.status, pe.msg
	}
	var ids []string
	for _, i := range s.gen.Instances() {
		if set, err := s.gen.ModelSet(i.ModelSetID); err == nil && set.Kind == generation.KindAce && i.Enabled {
			ids = append(ids, i.ID)
		}
	}
	return inst, http.StatusNotFound, "unknown model; available: " + strings.Join(ids, ", ")
}

// releaseTaskRequest is the official ACE-Step server's /release_task body,
// as far as acestep.cpp supports it.
type releaseTaskRequest struct {
	Prompt         string   `json:"prompt"`
	Caption        string   `json:"caption"`
	Lyrics         string   `json:"lyrics"`
	AudioDuration  float64  `json:"audio_duration"`
	BPM            int      `json:"bpm"`
	KeyScale       string   `json:"key_scale"`
	TimeSignature  string   `json:"time_signature"`
	VocalLanguage  string   `json:"vocal_language"`
	Seed           *int64   `json:"seed"`
	UseRandomSeed  *bool    `json:"use_random_seed"`
	InferenceSteps int      `json:"inference_steps"`
	GuidanceScale  float64  `json:"guidance_scale"`
	AudioFormat    string   `json:"audio_format"`
	Model          string   `json:"model"`
	Thinking       bool     `json:"thinking"`
	BatchSize      *float64 `json:"batch_size"`

	TaskType           string   `json:"task_type"`
	Track              string   `json:"track"`
	RepaintingStart    *float64 `json:"repainting_start"`
	RepaintingEnd      *float64 `json:"repainting_end"`
	AudioCoverStrength *float64 `json:"audio_cover_strength"`
	// Paths on the official server's disk; refused here, where a client's
	// paths mean nothing. Upload the files instead.
	SrcAudioPath       string `json:"src_audio_path"`
	ReferenceAudioPath string `json:"reference_audio_path"`
}

// readReleaseTask reads /release_task's body: JSON, or the official
// server's multipart form, with each field in a part of its own and the
// songs in src_audio (or ctx_audio) and reference_audio (or ref_audio).
func readReleaseTask(w http.ResponseWriter, r *http.Request) (releaseTaskRequest, *aceFile, *aceFile, int, error) {
	var in releaseTaskRequest
	ct := r.Header.Get("Content-Type")
	if !strings.HasPrefix(ct, "multipart/form-data") {
		if ct != "" && !strings.HasPrefix(ct, "application/json") {
			return in, nil, nil, http.StatusUnsupportedMediaType, errors.New("send the request as JSON or multipart/form-data")
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&in); err != nil {
			return in, nil, nil, http.StatusBadRequest, errors.New("the body is not valid JSON: " + err.Error())
		}
		return in, nil, nil, 0, nil
	}
	form, err := readMusicForm(w, r)
	if err != nil {
		return in, nil, nil, http.StatusBadRequest, err
	}
	defer form.RemoveAll()
	if err := formFields(form, &in); err != nil {
		return in, nil, nil, http.StatusBadRequest, err
	}
	src, ref, err := musicUploads(form, []string{"src_audio", "ctx_audio"}, []string{"reference_audio", "ref_audio"})
	if err != nil {
		return in, nil, nil, http.StatusBadRequest, err
	}
	return in, src, ref, 0, nil
}

// formFields sets each of dst's JSON-tagged fields from the form value of
// that name, as the official server reads its multipart fields.
func formFields(form *multipart.Form, dst any) error {
	v := reflect.ValueOf(dst).Elem()
	for i := 0; i < v.NumField(); i++ {
		name := strings.Split(v.Type().Field(i).Tag.Get("json"), ",")[0]
		raw := strings.TrimSpace(firstValue(form, name))
		if name == "" || raw == "" {
			continue
		}
		f := v.Field(i)
		if f.Kind() == reflect.Pointer {
			f.Set(reflect.New(f.Type().Elem()))
			f = f.Elem()
		}
		var err error
		switch f.Kind() {
		case reflect.String:
			f.SetString(raw)
		case reflect.Bool:
			var b bool
			b, err = strconv.ParseBool(raw)
			f.SetBool(b)
		case reflect.Int, reflect.Int64:
			var n int64
			n, err = strconv.ParseInt(raw, 10, 64)
			f.SetInt(n)
		case reflect.Float64:
			var x float64
			x, err = strconv.ParseFloat(raw, 64)
			f.SetFloat(x)
		}
		if err != nil {
			return fmt.Errorf("%s: %q is not a valid value", name, raw)
		}
	}
	return nil
}

func (s *Server) handleReleaseTask(w http.ResponseWriter, r *http.Request) {
	if !s.musicEnabled() {
		aceStepError(w, http.StatusNotFound, "music generation is not enabled on this server")
		return
	}
	in, src, ref, status, err := readReleaseTask(w, r)
	if err != nil {
		aceStepError(w, status, err.Error())
		return
	}
	if in.SrcAudioPath != "" || in.ReferenceAudioPath != "" {
		aceStepError(w, http.StatusBadRequest, "src_audio_path and reference_audio_path are not supported; upload the files as src_audio and reference_audio")
		return
	}
	if in.BatchSize != nil && *in.BatchSize > 1 {
		aceStepError(w, http.StatusBadRequest, "batch_size > 1 is not supported")
		return
	}
	req := musicRequest{
		Model: in.Model, Caption: firstNonEmpty(in.Prompt, in.Caption), Lyrics: in.Lyrics, Duration: in.AudioDuration,
		BPM: in.BPM, KeyScale: in.KeyScale, TimeSignature: in.TimeSignature, VocalLanguage: in.VocalLanguage,
		InferenceSteps: in.InferenceSteps, GuidanceScale: in.GuidanceScale, RewriteCaption: in.Thinking,
		OutputFormat: map[string]string{"wav": "wav", "flac": "wav"}[in.AudioFormat],
		TaskType:     in.TaskType, Track: in.Track, RepaintingStart: in.RepaintingStart, RepaintingEnd: in.RepaintingEnd,
		AudioCoverStrength: in.AudioCoverStrength,
		source:             src, ref: ref, SourceAudio: src != nil, ReferenceAudio: ref != nil,
	}
	if in.Seed != nil && (in.UseRandomSeed == nil || !*in.UseRandomSeed) {
		req.Seed = in.Seed
	}
	inst, status, detail := s.pickAceStepInstance(req.Model)
	if status != 0 {
		aceStepError(w, status, detail)
		return
	}
	if code, msg := s.checkMusic(inst, req, "prompt"); code != "" {
		aceStepError(w, http.StatusBadRequest, msg)
		return
	}
	job, err := s.submitMusic(inst, req)
	if errors.Is(err, generation.ErrQueueFull) {
		aceStepError(w, http.StatusTooManyRequests, err.Error())
		return
	}
	if err != nil {
		aceStepError(w, http.StatusBadRequest, err.Error())
		return
	}
	view, _ := s.genMusic.view(job.ID)
	aceStepOK(w, map[string]any{"task_id": view.ID, "status": view.Status, "queue_position": view.Position})
}

// aceStepStatus is the official server's numeric status: 0 queued or
// running, 1 succeeded, 2 failed.
func aceStepStatus(status string) int {
	switch status {
	case musicSucceeded:
		return 1
	case musicFailed, musicCancelled:
		return 2
	}
	return 0
}

func (s *Server) handleQueryResult(w http.ResponseWriter, r *http.Request) {
	if !s.musicEnabled() {
		aceStepError(w, http.StatusNotFound, "music generation is not enabled on this server")
		return
	}
	var in struct {
		TaskIDList []string `json:"task_id_list"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&in); err != nil {
		aceStepError(w, http.StatusBadRequest, "the body is not valid JSON: "+err.Error())
		return
	}
	out := []map[string]any{}
	for _, id := range in.TaskIDList {
		j, ok := s.genMusic.view(id)
		if !ok {
			out = append(out, map[string]any{"task_id": id, "status": 2, "result": "[]"})
			continue
		}
		st := aceStepStatus(j.Status)
		item := map[string]any{
			"file": "", "wave": "", "status": st, "create_time": j.Submitted.Unix(),
			"metas": map[string]any{
				"bpm": j.Result["bpm"], "duration": j.Result["duration"], "genres": "",
				"keyscale": j.Result["keyscale"], "timesignature": j.Result["timesignature"],
			},
			"prompt": firstNonEmpty(asString(j.Result["caption"]), j.Request.Caption), "lyrics": firstNonEmpty(asString(j.Result["lyrics"]), j.Request.Lyrics),
			"seed_value": asString(j.Result["seed"]), "lm_model": j.lmModel, "dit_model": j.ditModel,
			"generation_info": j.Error, "env": "",
		}
		if st == 1 {
			item["file"] = "/v1/audio?path=" + j.ID + "." + j.Request.ext()
		}
		raw, _ := json.Marshal([]any{item})
		out = append(out, map[string]any{"task_id": j.ID, "status": st, "result": string(raw)})
	}
	aceStepOK(w, out)
}

func asString(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	}
	return fmt.Sprint(v)
}

// musicFileRE is the only shape of path /v1/audio serves, so a path can
// never reach another file.
var musicFileRE = regexp.MustCompile(`^[0-9a-f]{16}\.(mp3|wav)$`)

func (s *Server) handleAceStepAudio(w http.ResponseWriter, r *http.Request) {
	if !s.musicEnabled() {
		aceStepError(w, http.StatusNotFound, "music generation is not enabled on this server")
		return
	}
	name := path.Base(r.URL.Query().Get("path"))
	if !musicFileRE.MatchString(name) {
		aceStepError(w, http.StatusBadRequest, "path is not a music file from this server")
		return
	}
	file := filepath.Join(s.musicDir(), name)
	if _, err := os.Stat(file); err != nil {
		aceStepError(w, http.StatusNotFound, "no such audio file")
		return
	}
	serveMusicFile(w, r, file)
}
