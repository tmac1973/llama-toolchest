package forks

import (
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

func TestCuratedForksAreComplete(t *testing.T) {
	idOK := regexp.MustCompile(`^[a-z0-9-]+$`)
	shaOK := regexp.MustCompile(`^[0-9a-f]{40}$`)
	seen := map[string]bool{}
	for _, f := range Curated() {
		if !idOK.MatchString(f.ID) || seen[f.ID] {
			t.Errorf("fork ID %q is not a unique lowercase slug", f.ID)
		}
		seen[f.ID] = true
		if !shaOK.MatchString(f.PinnedSHA) {
			t.Errorf("%s: pinned SHA %q is not a full commit SHA", f.ID, f.PinnedSHA)
		}
		if len(f.Backends) == 0 || len(f.ModelFiles) == 0 || f.ModelRepo == "" {
			t.Errorf("%s: needs backends, model files and a model repo", f.ID)
		}
		if !shaOK.MatchString(f.ModelRevision) {
			t.Errorf("%s: model revision %q is not a full commit SHA", f.ID, f.ModelRevision)
		}
		paths := map[string]bool{}
		for _, m := range f.ModelFiles {
			if m.Bytes <= 0 || paths[m.Path] {
				t.Errorf("%s: file %q has no size or is listed twice", f.ID, m.Path)
			}
			paths[m.Path] = true
		}
		if !paths[f.Recipe.ModelFileArg] {
			t.Errorf("%s: model file %q is not in the model files", f.ID, f.Recipe.ModelFileArg)
		}
		if f.Recipe.DraftFileArg != "" && !paths[f.Recipe.DraftFileArg] {
			t.Errorf("%s: draft file %q is not in the model files", f.ID, f.Recipe.DraftFileArg)
		}
		if err := CheckExtraArgs(f.Recipe.Args); err != nil {
			t.Errorf("%s: recipe sets an option toolchest sets: %v", f.ID, err)
		}
		if err := CheckEnv(f.Recipe.Env); err != nil {
			t.Errorf("%s: recipe env: %v", f.ID, err)
		}
	}
}

func TestAvalonModelSet(t *testing.T) {
	f, ok := Find("avalon")
	if !ok {
		t.Fatal("avalon not curated")
	}
	var total int64
	shards := 0
	for _, m := range f.ModelFiles {
		total += m.Bytes
		if strings.Contains(m.Path, "-IQ3KS-") {
			shards++
		}
	}
	if shards != 33 {
		t.Errorf("%d shards, want 33", shards)
	}
	if gib := float64(total) / (1 << 30); gib < 89 || gib > 90.5 {
		t.Errorf("model set is %.1f GiB, want about 89.7", gib)
	}
	for _, want := range []string{"moe-cache-ranking.bin", f.Recipe.DraftFileArg} {
		if !slices.ContainsFunc(f.ModelFiles, func(m ModelFile) bool { return m.Path == want }) {
			t.Errorf("%s missing from the model files", want)
		}
	}
	if _, ok := Find("nope"); ok {
		t.Error("Find found an unknown fork")
	}
}

func TestStoreRoundTrip(t *testing.T) {
	dir := t.TempDir()
	s, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(StorePath(dir)); !os.IsNotExist(err) {
		t.Error("an empty store wrote a file")
	}
	set := ModelSet{ForkID: "avalon", Dir: "/models/fork-models/x", Complete: true}
	inst := Instance{ID: "avalon-qwen", ForkID: "avalon", GPU: 1, Port: 8082,
		ExtraArgs: []string{"-c", "32768"}, ExtraEnv: []string{"GGML_CUDA_MOE_PIN=0"}, AutoStart: true}
	if err := s.PutModelSet(set); err != nil {
		t.Fatal(err)
	}
	if err := s.AddInstance(inst); err != nil {
		t.Fatal(err)
	}

	s2, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := s2.ModelSet("avalon"); err != nil || got != set {
		t.Errorf("model set = %+v, %v", got, err)
	}
	got, err := s2.Instance("avalon-qwen")
	if err != nil || got.GPU != 1 || got.Port != 8082 || !got.AutoStart ||
		!slices.Equal(got.ExtraArgs, inst.ExtraArgs) || !slices.Equal(got.ExtraEnv, inst.ExtraEnv) {
		t.Errorf("instance = %+v, %v", got, err)
	}

	// A copy handed out does not share the store's memory.
	got.ExtraArgs[0] = "changed"
	if again, _ := s2.Instance("avalon-qwen"); again.ExtraArgs[0] != "-c" {
		t.Error("changing a returned instance changed the store")
	}

	inst.Port = 8083
	if err := s2.UpdateInstance(inst); err != nil {
		t.Fatal(err)
	}
	if err := s2.DeleteInstance("avalon-qwen"); err != nil {
		t.Fatal(err)
	}
	if err := s2.DeleteModelSet("avalon"); err != nil {
		t.Fatal(err)
	}
	if _, err := s2.Instance("avalon-qwen"); !errors.Is(err, ErrNotFound) {
		t.Errorf("deleted instance: %v", err)
	}
	if err := s2.DeleteModelSet("avalon"); !errors.Is(err, ErrNotFound) {
		t.Errorf("deleting a missing set: %v", err)
	}
}

func TestStoreRefusesBadInstances(t *testing.T) {
	s, err := Load(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	ok := Instance{ID: "avalon-qwen", ForkID: "avalon"}
	if err := s.AddInstance(ok); err != nil {
		t.Fatal(err)
	}
	if err := s.AddInstance(ok); !errors.Is(err, ErrIDTaken) {
		t.Errorf("duplicate ID: %v", err)
	}
	for name, inst := range map[string]Instance{
		"empty ID":        {ForkID: "avalon"},
		"upper case ID":   {ID: "Avalon", ForkID: "avalon"},
		"space in ID":     {ID: "a b", ForkID: "avalon"},
		"unknown fork":    {ID: "x", ForkID: "nope"},
		"port in args":    {ID: "x", ForkID: "avalon", ExtraArgs: []string{"--port=9000"}},
		"model in args":   {ID: "x", ForkID: "avalon", ExtraArgs: []string{"-m", "other.gguf"}},
		"env without =":   {ID: "x", ForkID: "avalon", ExtraEnv: []string{"FOO"}},
		"lower case env":  {ID: "x", ForkID: "avalon", ExtraEnv: []string{"foo=1"}},
		"digit-first env": {ID: "x", ForkID: "avalon", ExtraEnv: []string{"1FOO=1"}},
	} {
		if err := s.AddInstance(inst); err == nil {
			t.Errorf("%s: saved", name)
		}
	}
	if err := s.UpdateInstance(Instance{ID: "missing", ForkID: "avalon"}); !errors.Is(err, ErrNotFound) {
		t.Errorf("updating a missing instance: %v", err)
	}
}

func TestLoadRefusesOtherSchema(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "config"), 0o755)
	os.WriteFile(StorePath(dir), []byte(`{"schema_version": 2}`), 0o600)
	if _, err := Load(dir); err == nil {
		t.Error("loaded a store of another schema version")
	}
}

func TestArgs(t *testing.T) {
	f, _ := Find("avalon")
	set := ModelSet{ForkID: "avalon", Dir: "/m/avalon"}
	inst := Instance{ID: "q", ForkID: "avalon", Port: 8085, ExtraArgs: []string{"-c", "32768"}}
	args := Args(f, set, inst)
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"--host 127.0.0.1 --port 8085",
		"-m /m/avalon/" + f.Recipe.ModelFileArg,
		"-md /m/avalon/" + f.Recipe.DraftFileArg,
		"--spec-type draft-mtp",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("args lack %q: %s", want, joined)
		}
	}
	if !slices.Equal(args[len(args)-2:], []string{"-c", "32768"}) {
		t.Errorf("extra arguments are not last: %s", joined)
	}

	f.Recipe.DraftFileArg = ""
	if slices.Contains(Args(f, set, inst), "-md") {
		t.Error("-md given with no draft file")
	}
}

func TestEnv(t *testing.T) {
	f, _ := Find("avalon")
	f.Recipe.Env = []string{"GGML_CUDA_MOE_PIN=1", "A=1"}
	inst := Instance{ExtraEnv: []string{"GGML_CUDA_MOE_PIN=0", "B=2"}}
	got := Env(f, inst, "rocm", 1)
	want := []string{"HIP_VISIBLE_DEVICES=1", "GGML_CUDA_MOE_PIN=0", "A=1", "B=2"}
	if !slices.Equal(got, want) {
		t.Errorf("rocm env = %v, want %v", got, want)
	}
	if got := Env(f, Instance{}, "cuda", 0); got[0] != "CUDA_VISIBLE_DEVICES=0" {
		t.Errorf("cuda env = %v", got)
	}
	// An instance can also replace the GPU selector.
	if got := Env(f, Instance{ExtraEnv: []string{"HIP_VISIBLE_DEVICES=0"}}, "rocm", 1); got[0] != "HIP_VISIBLE_DEVICES=0" {
		t.Errorf("selector not replaced: %v", got)
	}
}

func TestCheckNeeds(t *testing.T) {
	f, _ := Find("avalon")
	good := HostInfo{RAMGiB: 64, AVX2Known: true, AVX2: true, GPUArch: "gfx1100", Backend: "rocm", FreeDiskGiB: 200}
	if got := CheckNeeds(f, good); len(got) != 0 {
		t.Errorf("a host that meets all needs got %v", got)
	}
	if got := CheckNeeds(f, HostInfo{}); len(got) != 0 {
		t.Errorf("an unknown host got %v", got)
	}
	for name, c := range map[string]struct {
		host HostInfo
		word string
	}{
		"little RAM":   {HostInfo{RAMGiB: 24}, "24 GiB of RAM"},
		"no AVX2":      {HostInfo{AVX2Known: true}, "AVX2"},
		"other arch":   {HostInfo{Backend: "rocm", GPUArch: "gfx1201"}, "gfx1201"},
		"vulkan":       {HostInfo{Backend: "vulkan"}, "vulkan"},
		"little space": {HostInfo{FreeDiskGiB: 50}, "50 GiB free"},
	} {
		got := CheckNeeds(f, c.host)
		if len(got) != 1 || !strings.Contains(got[0], c.word) {
			t.Errorf("%s: %v, want one sentence with %q", name, got, c.word)
		}
	}
	if got := CheckNeeds(f, HostInfo{Backend: "cuda"}); len(got) != 0 {
		t.Errorf("cuda host: %v", got)
	}
}
