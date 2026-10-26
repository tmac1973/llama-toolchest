package generation

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// LaunchOptions adjusts a launch for special runs.
type LaunchOptions struct {
	// MaxVRAMMiB, when positive, is sd-server's --max-vram instead of the
	// instance's reservation: a measured test run uses the room the card
	// can give the instance. ace-server takes no memory limit.
	MaxVRAMMiB int
}

// deviceEnvVar is the variable that limits a backend process to one GPU,
// by build profile backend.
var deviceEnvVar = map[string]string{
	"rocm":   "HIP_VISIBLE_DEVICES",
	"cuda":   "CUDA_VISIBLE_DEVICES",
	"vulkan": "GGML_VK_VISIBLE_DEVICES",
}

// sdSlotFlags maps a model set slot to sd-server's flag for it.
var sdSlotFlags = map[string]string{
	"diffusion_model": "--diffusion-model",
	"model":           "--model",
	"llm":             "--llm",
	"llm_vision":      "--llm_vision",
	"clip_l":          "--clip_l",
	"t5xxl":           "--t5xxl",
	"tokenizer":       "--tokenizer",
	"vae":             "--vae",
	"tae":             "--tae",
	// Video sets.
	"high_noise_diffusion_model": "--high-noise-diffusion-model",
	"embeddings_connectors":      "--embeddings-connectors",
	"audio_vae":                  "--audio-vae",
	"clip_vision":                "--clip_vision",
}

// BuildBackend returns the profile backend ("rocm", "cuda", "vulkan") a
// build was made with.
func BuildBackend(b builder.BuildResult) (string, error) {
	prof, ok := builder.FindProfile(b.Profile)
	if !ok {
		return "", fmt.Errorf("build %s has unknown profile %q", b.ID, b.Profile)
	}
	return prof.Backend, nil
}

// CheckGPUVendor reports whether a build can run on the machine's GPUs:
// a rocm build needs AMD GPUs, a cuda build NVIDIA ones; a vulkan build
// runs on either. monitorVendor is monitor.GPUVendor()'s answer.
func CheckGPUVendor(backend, monitorVendor string) error {
	need := map[string]string{"rocm": "rocm", "cuda": "nvidia"}[backend]
	if need == "" || monitorVendor == "" || monitorVendor == need {
		return nil
	}
	gpus := map[string]string{"rocm": "AMD", "nvidia": "NVIDIA"}
	return fmt.Errorf("a %s build needs %s GPUs, but this machine's GPUs are %s", backend, gpus[need], gpus[monitorVendor])
}

// InstanceDir is where an instance keeps its own files.
func InstanceDir(dataDir, id string) string {
	return filepath.Join(dataDir, "gen-instances", id)
}

// LaunchSpec returns how to run an instance: its program, arguments,
// environment and health URL. host is the address it listens on (the same
// one the router uses). For an ace instance it also (re)creates the
// directory of links to the set's files that ace-server scans.
func LaunchSpec(inst Instance, set ModelSet, build builder.BuildResult, host, dataDir string, opts LaunchOptions) (process.SupervisorConfig, error) {
	var cfg process.SupervisorConfig
	if !set.Complete {
		return cfg, fmt.Errorf("model set %s is not fully downloaded", set.ID)
	}
	if inst.Port == 0 {
		return cfg, fmt.Errorf("instance %s has no port", inst.ID)
	}
	backend, err := BuildBackend(build)
	if err != nil {
		return cfg, err
	}
	envVar := deviceEnvVar[backend]
	if envVar == "" {
		return cfg, fmt.Errorf("build %s uses the %s profile; generation backends need rocm, cuda or vulkan", build.ID, backend)
	}
	port := strconv.Itoa(inst.Port)
	base := fmt.Sprintf("http://127.0.0.1:%d", inst.Port)
	if host == "" {
		host = "0.0.0.0"
	}

	var args []string
	switch set.Kind {
	case KindSD, KindVideo:
		args = []string{"--listen-ip", host, "--listen-port", port}
		for _, c := range set.Components {
			flag := sdSlotFlags[c.Slot]
			if flag == "" {
				return cfg, fmt.Errorf("model set %s: sd-server has no flag for slot %q", set.ID, c.Slot)
			}
			args = append(args, flag, c.Path)
		}
		args = append(args, set.RunArgs()...)
		args = append(args, inst.ExtraArgs...)
		switch {
		case opts.MaxVRAMMiB > 0:
			args = append(args, "--max-vram", maxVRAMGiB(opts.MaxVRAMMiB))
		case inst.ReservedMiB > 0:
			args = append(args, "--max-vram", maxVRAMGiB(inst.ReservedMiB))
		}
		cfg.Name, cfg.HealthURL = "sd-server:"+inst.ID, base+"/v1/models"
	case KindAce:
		modelsDir := filepath.Join(InstanceDir(dataDir, inst.ID), "models")
		if err := linkAceModels(modelsDir, set); err != nil {
			return cfg, err
		}
		// No --keep-loaded: loading one part at a time peaked at about
		// 7.2 GB of VRAM and idled at 0.3 GB on the R9700, against about
		// 20 GB held with --keep-loaded, for the same speed.
		args = []string{"--host", host, "--port", port, "--models", modelsDir, "--max-batch", "1"}
		args = append(args, inst.ExtraArgs...)
		cfg.Name, cfg.HealthURL = "ace-server:"+inst.ID, base+"/health"
	default:
		return cfg, fmt.Errorf("model set %s has unknown kind %q", set.ID, set.Kind)
	}

	env := process.BuildEnv(filepath.Dir(build.BinaryPath), nil)
	env = setEnv(env, envVar, strconv.Itoa(models.BackendDeviceIndex(backend, inst.GPU)))

	cfg.BinaryPath = build.BinaryPath
	cfg.Args = args
	cfg.Env = env
	cfg.WorkDir = InstanceDir(dataDir, inst.ID)
	if err := os.MkdirAll(cfg.WorkDir, 0o755); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// maxVRAMGiB renders a reservation for sd-server's --max-vram: GiB, rounded
// down to one decimal place, so sd.cpp plans inside the reservation.
func maxVRAMGiB(mib int) string {
	tenths := mib * 10 / 1024
	return fmt.Sprintf("%d.%d", tenths/10, tenths%10)
}

// setEnv sets key=value, replacing any inherited value. The device
// selector must not follow BuildEnv's "inherited wins" rule: a stray
// HIP_VISIBLE_DEVICES around toolchest would put the backend on the wrong
// GPU.
func setEnv(env []string, key, value string) []string {
	out := env[:0:0]
	for _, kv := range env {
		if !strings.HasPrefix(kv, key+"=") {
			out = append(out, kv)
		}
	}
	return append(out, key+"="+value)
}

// linkAceModels fills dir with links to an ACE-Step set's four files, and
// nothing else: ace-server loads whatever GGUF files its --models directory
// holds.
func linkAceModels(dir string, set ModelSet) error {
	if err := os.RemoveAll(dir); err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for _, c := range set.Components {
		if c.Path == "" {
			return fmt.Errorf("model set %s: %s file missing", set.ID, c.Slot)
		}
		if err := os.Symlink(c.Path, filepath.Join(dir, filepath.Base(c.Path))); err != nil {
			return fmt.Errorf("linking %s: %w", c.Slot, err)
		}
	}
	return nil
}
