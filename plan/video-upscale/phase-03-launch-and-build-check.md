# Phase 03 — Launch vupscale instances and check the build

**Depends on:** Phase 02 (`KindVUpscale`, the slots, `ProjectForKind`) · **Enables:** Phase 04 (jobs need a running instance), Phase 06 (Measure starts the instance)

## Goal

A vupscale instance starts sd-server with no diffusion model, with its model folders, its `--max-vram` and a working folder for job input and output. A build without DGL (older than stable-diffusion.cpp master-1116) is refused before the start, with a message that names the build it needs. After this phase a vupscale instance starts, shows as running and passes its health check, but it has no jobs yet.

## Files touched

- `internal/generation/launch.go`:
  - `LaunchSpec`: replace the Phase 02 error with the real `KindVUpscale` case.
  - `buildArgs`: a `KindVUpscale` case.
  - The launch params struct gets `ModelDirs []string`.
- `internal/generation/build_flags.go` — `BuildHasDGL(build builder.BuildResult) bool`, using `buildHelp` and `helpLists(help, "--dgl-dir")`.
- `internal/generation/vupscale.go` — new:
  - `VUpscaleWorkDirs(dataDir, instID string) (in, out string, err error)`, which creates and returns `<instance dir>/input` and `<instance dir>/output` (step 5).
  - `const VUpscaleMinBuild = "master-1116"`.
- `internal/api/gen_instances.go` — the instance form and list show the build error for a vupscale instance whose build has no DGL.
- `docs/guides/generation.md` — one line: video upscaler sets need stable-diffusion.cpp master-1116 or newer, with the exact error text.
- Tests: `internal/generation/launch_test.go`, `build_flags_test.go`, `internal/api/gen_instances_test.go`.

## Steps

1. **Build check.**
   - `BuildHasDGL` returns true when the `--help` text lists `--dgl-dir`.
   - When the help text is empty (the probe failed), it returns true and logs a warning. This matches how `adaptArgsForBuild` treats a failed probe: the instance tries to start, and sd.cpp's own error shows if DGL is missing.
   - In `LaunchSpec`, for `KindVUpscale`, after the build is resolved: when `!BuildHasDGL(build)`, return the error "video upscaler sets need stable-diffusion.cpp master-1116 or newer (DGL); build <build.ID> is older. Build a newer stable-diffusion.cpp on the Builds page, or pick a newer build for this instance".
2. **Model folders.** DGL `load_dit` and `load_vae` take the files by path. The script uses the absolute paths of the set's two files (Phase 04). Pass the folder that holds each file as `--model-dir`, de-duplicated and in slot order (dit, then vae). This keeps the server's model-folder rule satisfied for both catalog and custom sets, wherever under the models folder the files are.
3. **Launch args** (`buildArgs`, `KindVUpscale`):
   ```go
   add("--listen-ip", p.Host)
   add("--listen-port", p.Port)
   for _, d := range p.ModelDirs { add("--model-dir", d) }
   if p.MaxVRAM != "" { add("--max-vram", p.MaxVRAM) }
   args = append(args, set.RunArgs()...)
   args = append(args, extra...)
   ```
   - Leave out `--dgl-dir`. The app never saves scripts on the server. It sends the source with each job.
   - Set `p.MaxVRAM` exactly as the video case does: the Measure room, else `BudgetMiB`, else `ReservedMiB`. sd.cpp's DGL context uses `--max-vram` (upstream `docs/dgl/README.md`).
   - Do not call `adaptArgsForBuild` for vupscale: it only drops high-noise flags, which vupscale sets never have.
   - `cfg.Name = "sd-server:"+inst.ID` and `cfg.HealthURL = base+"/v1/models"`, as for upscale.
4. **Environment.** `sdServerEnv` already adds `GGML_CUDA_REUSE_BUFFERS=0` for every sd kind except ace, so vupscale gets it with no change. Add a test that pins this.
5. **Work folders.** `VUpscaleWorkDirs` also creates both folders (`os.MkdirAll`, 0o755) and returns an error if it cannot. Every caller that writes there (`LaunchSpec` here, the job route in Phase 05, Measure in Phase 06) calls it first, so the folders exist even for an instance that never started. In `LaunchSpec`, call it before the start. sd-server runs with `cfg.WorkDir` = the instance folder, so a script can use the relative paths `input/<job>.<ext>` and `output/<job>.<ext>`. This matches the server guide's advice to write results under `output/` in the working folder.
6. **Instance UI.** In the instance list, a vupscale instance whose resolved build fails `BuildHasDGL` shows the same error text under its name, as other start errors do. The Start button stays enabled, so a later build can fix it without editing the instance.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`

## Test plan

- `buildArgs` golden test for a vupscale set with a reservation of 6 GiB, two files in different folders and one extra arg. The result is `--listen-ip … --listen-port … --model-dir <dit dir> --model-dir <vae dir> --max-vram 6 <extra>`.
- The same set with both files in one folder gives one `--model-dir`.
- `BuildHasDGL`:
  - a fake binary whose `--help` lists `--dgl-dir --steps` → true;
  - one that lists only `--steps` → false;
  - a binary that fails → true, with a logged warning.
- `LaunchSpec` with the old-build fake returns the error text from step 1.
- `LaunchSpec` creates `input/` and `output/` in the instance folder.
- Manual on asmodean: build stable-diffusion.cpp master-1122 or newer (ROCm 10). Add a RelayVSR ×2 instance and set a hand reservation of 16 GiB on the Server page. Like every generation instance, it cannot start without a reservation, and Measure for vupscale comes only in Phase 06. Start it. It reaches "running", `GET http://127.0.0.1:<port>/v1/models` answers, and the process args match the golden test. Pin the instance to an older build (for example 1106): Start shows the step 1 message.

## Commit

`feat(generation): start video upscaler instances and refuse builds without DGL`

## Rollback

Revert the commit. Phase 02's "not supported yet" error comes back, and vupscale instances stop starting. No data changes.
