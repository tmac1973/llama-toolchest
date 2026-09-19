package models

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// profileRegistry returns a registry with one installed model.
func profileRegistry(t *testing.T) (*Registry, string) {
	t.Helper()
	reg, _ := pendingRegistry(t)
	m := &Model{ID: "org--r--m-Q4_K_M", ModelID: "org/r-GGUF", Filename: "m-Q4_K_M.gguf", Quant: "Q4_K_M", FilePath: "/nowhere/m-Q4_K_M.gguf"}
	if err := reg.Add(m); err != nil {
		t.Fatal(err)
	}
	return reg, m.ID
}

func setContext(t *testing.T, reg *Registry, id string, ctx int) {
	t.Helper()
	cfg, err := reg.GetConfig(id)
	if err != nil {
		t.Fatal(err)
	}
	next := cloneConfig(*cfg)
	next.ActiveProfile = "" // what the config form posts
	next.ContextSize = ctx
	if err := reg.SetConfig(id, &next); err != nil {
		t.Fatal(err)
	}
}

func TestProfileSaveListApplyDelete(t *testing.T) {
	reg, id := profileRegistry(t)
	setContext(t, reg, id, 16384)

	if replaced, err := reg.SaveProfile(id, "  Long   Ctx ", ProfileSourceUser, "b1"); err != nil || replaced {
		t.Fatalf("SaveProfile = %v, %v; want a new profile", replaced, err)
	}
	if name, edited := reg.ActiveProfileState(id); name != "Long Ctx" || edited {
		t.Fatalf("after save: active = %q edited=%v, want \"Long Ctx\" unedited", name, edited)
	}

	// An autosave that does not post the label keeps it, and the change
	// shows up as an edit.
	setContext(t, reg, id, 4096)
	if name, edited := reg.ActiveProfileState(id); name != "Long Ctx" || !edited {
		t.Fatalf("after edit: active = %q edited=%v, want \"Long Ctx\" edited", name, edited)
	}

	// Aliases and Enabled are the live model's, not the profile's.
	cfg, _ := reg.GetConfig(id)
	cfg.Aliases = []string{"mine"}
	if err := reg.ApplyProfile(id, "long ctx"); err != nil {
		t.Fatal(err)
	}
	cfg, _ = reg.GetConfig(id)
	if cfg.ContextSize != 16384 || !cfg.Enabled || len(cfg.Aliases) != 1 || cfg.ActiveProfile != "Long Ctx" {
		t.Fatalf("after apply: ctx=%d enabled=%v aliases=%v active=%q", cfg.ContextSize, cfg.Enabled, cfg.Aliases, cfg.ActiveProfile)
	}
	if _, edited := reg.ActiveProfileState(id); edited {
		t.Error("config reports edited right after restoring its profile")
	}

	if got := reg.Profiles(id); len(got) != 1 || got[0].Source != ProfileSourceUser || got[0].BuildID != "b1" {
		t.Fatalf("Profiles = %+v", got)
	}
	if err := reg.DeleteProfile(id, "LONG CTX"); err != nil {
		t.Fatal(err)
	}
	cfg, _ = reg.GetConfig(id)
	if cfg.ActiveProfile != "" || cfg.ContextSize != 16384 {
		t.Errorf("after deleting the active profile: active=%q ctx=%d; want label cleared, config kept", cfg.ActiveProfile, cfg.ContextSize)
	}
	if err := reg.DeleteProfile(id, "Long Ctx"); !errors.Is(err, ErrProfileNotFound) {
		t.Errorf("second delete err = %v, want ErrProfileNotFound", err)
	}
}

func TestProfileSaveOverFoldedNameReplaces(t *testing.T) {
	reg, id := profileRegistry(t)
	if _, err := reg.SaveProfile(id, "Long Ctx", ProfileSourceUser, ""); err != nil {
		t.Fatal(err)
	}
	setContext(t, reg, id, 32768)
	replaced, err := reg.SaveProfile(id, "long  ctx", ProfileSourceUser, "")
	if err != nil || !replaced {
		t.Fatalf("SaveProfile = %v, %v; want replaced", replaced, err)
	}
	got := reg.Profiles(id)
	if len(got) != 1 || got[0].Config.ContextSize != 32768 || got[0].Name != "long ctx" {
		t.Fatalf("Profiles = %+v; want one entry holding the new config under the new spelling", got)
	}
}

func TestProfileNameRules(t *testing.T) {
	reg, id := profileRegistry(t)
	if _, err := reg.SaveProfile(id, "   ", ProfileSourceUser, ""); !errors.Is(err, ErrProfileName) {
		t.Errorf("blank name err = %v, want ErrProfileName", err)
	}
	if _, err := reg.SaveProfile(id, strings.Repeat("x", MaxProfileNameLen+1), ProfileSourceUser, ""); err == nil {
		t.Error("over-long name accepted")
	}
}

// The stored profile must not share memory with the live config: editing
// a sampling pointer in place afterwards must not change the profile.
func TestProfileDoesNotAliasLiveConfig(t *testing.T) {
	reg, id := profileRegistry(t)
	cfg, _ := reg.GetConfig(id)
	temp := 0.7
	cfg.Temperature = &temp
	if _, err := reg.SaveProfile(id, "p", ProfileSourceUser, ""); err != nil {
		t.Fatal(err)
	}
	*cfg.Temperature = 1.5
	p, err := reg.GetProfile(id, "p")
	if err != nil {
		t.Fatal(err)
	}
	if *p.Config.Temperature != 0.7 {
		t.Errorf("profile temperature = %v after editing the live config, want 0.7", *p.Config.Temperature)
	}
}

// Profiles are filed by repository and file, so they survive the model
// being deleted and registered again under a different registry ID (a
// disk scan builds IDs differently from a download).
func TestProfilesSurviveDeleteAndReRegister(t *testing.T) {
	reg, id := profileRegistry(t)
	if _, err := reg.SaveProfile(id, "keep", ProfileSourceUser, ""); err != nil {
		t.Fatal(err)
	}
	if err := reg.Remove(id); err != nil {
		t.Fatal(err)
	}
	if n := len(reg.AllProfiles()); n != 1 {
		t.Fatalf("profiles after removing the model = %d, want 1", n)
	}
	again := &Model{ID: "org--r-GGUF--m-Q4_K_M", ModelID: "org/r-GGUF", Filename: "m-Q4_K_M.gguf", Quant: "Q4_K_M"}
	if err := reg.Add(again); err != nil {
		t.Fatal(err)
	}
	if got := reg.Profiles(again.ID); len(got) != 1 || got[0].Name != "keep" {
		t.Fatalf("Profiles after re-registering = %+v, want the saved profile", got)
	}
}

func TestProfileReadOnlyRegistryRefuses(t *testing.T) {
	dataDir := t.TempDir()
	writeRegistryFile(t, dataDir, `{"schema_version": 99, "models": {`+keptModelJSON+`}, "configs": {"kept--m.gguf": {}}}`)
	reg := NewRegistry(dataDir, filepath.Join(dataDir, "models"))
	if _, err := reg.SaveProfile("kept--m.gguf", "p", ProfileSourceUser, ""); !errors.Is(err, ErrRegistryReadOnly) {
		t.Errorf("SaveProfile err = %v, want ErrRegistryReadOnly", err)
	}
	if _, err := reg.ImportProfile(ConfigProfile{RepoID: "a/b", Filename: "c.gguf", Name: "p"}); !errors.Is(err, ErrRegistryReadOnly) {
		t.Errorf("ImportProfile err = %v, want ErrRegistryReadOnly", err)
	}
}

// A schema-1 file (no profiles) loads, and is written back as the current
// version with the profile list.
func TestProfilesUpgradeSchemaOneFile(t *testing.T) {
	dataDir := t.TempDir()
	path := writeRegistryFile(t, dataDir, `{"schema_version": 1, "models": {`+keptModelJSON+`}, "configs": {"kept--m.gguf": {"context_size": 4096}}}`)
	reg := NewRegistry(dataDir, filepath.Join(dataDir, "models"))
	if reason := reg.ReadOnlyReason(); reason != "" {
		t.Fatalf("schema 1 file made the registry read-only: %s", reason)
	}
	if _, err := reg.SaveProfile("kept--m.gguf", "p", ProfileSourceUser, ""); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(path)
	var onDisk struct {
		SchemaVersion int             `json:"schema_version"`
		Profiles      []ConfigProfile `json:"config_profiles"`
	}
	if err := json.Unmarshal(raw, &onDisk); err != nil {
		t.Fatal(err)
	}
	if onDisk.SchemaVersion != 2 || len(onDisk.Profiles) != 1 || onDisk.Profiles[0].RepoID != "org/kept-GGUF" {
		t.Errorf("on disk: version %d, profiles %+v", onDisk.SchemaVersion, onDisk.Profiles)
	}
}

// setNonZero puts a non-zero value in v, whatever its kind.
func setNonZero(t *testing.T, v reflect.Value) {
	t.Helper()
	switch v.Kind() {
	case reflect.Bool:
		v.SetBool(true)
	case reflect.Int, reflect.Int64:
		v.SetInt(7)
	case reflect.Float64:
		v.SetFloat(0.5)
	case reflect.String:
		v.SetString("x")
	case reflect.Slice:
		s := reflect.MakeSlice(v.Type(), 1, 1)
		setNonZero(t, s.Index(0))
		v.Set(s)
	case reflect.Ptr:
		p := reflect.New(v.Type().Elem())
		setNonZero(t, p.Elem())
		v.Set(p)
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			setNonZero(t, v.Field(i))
		}
	default:
		t.Fatalf("setNonZero: unhandled kind %s", v.Kind())
	}
}

// Every ModelConfig field except the identity fields must count as a
// difference between profiles — a field added later is covered without
// anyone remembering this test.
func TestProfileEqualSeesEveryField(t *testing.T) {
	ignored := map[string]bool{"Enabled": true, "Aliases": true, "ActiveProfile": true}
	typ := reflect.TypeOf(ModelConfig{})
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		var changed ModelConfig
		setNonZero(t, reflect.ValueOf(&changed).Elem().Field(i))
		equal := ProfileEqual(ModelConfig{}, changed)
		if ignored[f.Name] && !equal {
			t.Errorf("%s: a difference counts, but it should be ignored", f.Name)
		}
		if !ignored[f.Name] && equal {
			t.Errorf("%s: a difference is not seen", f.Name)
		}
	}
}

// cloneConfig must copy what every pointer and slice field points to.
func TestCloneConfigCopiesEveryReference(t *testing.T) {
	var orig ModelConfig
	setNonZero(t, reflect.ValueOf(&orig).Elem())
	clone := cloneConfig(orig)
	ov, cv := reflect.ValueOf(orig), reflect.ValueOf(clone)
	for i := 0; i < ov.NumField(); i++ {
		name := ov.Type().Field(i).Name
		switch ov.Field(i).Kind() {
		case reflect.Ptr, reflect.Slice:
			if ov.Field(i).Pointer() == cv.Field(i).Pointer() {
				t.Errorf("%s: clone shares memory with the original", name)
			}
		case reflect.Map:
			t.Errorf("%s: map field — teach cloneConfig to copy it", name)
		}
	}
	if !reflect.DeepEqual(orig, clone) {
		t.Error("clone differs from the original")
	}
}
