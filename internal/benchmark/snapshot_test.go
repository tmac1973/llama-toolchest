package benchmark

import (
	"reflect"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Every ConfigSnapshot field that has a same-named ModelConfig field must
// be copied by SnapshotFromConfig, and every other field must be one this
// test knows about. A field added to the snapshot later therefore cannot
// be forgotten in the builder.
func TestSnapshotFromConfigCopiesEveryMatchingField(t *testing.T) {
	notFromConfig := map[string]bool{"ProfileName": true, "ProfileEdited": true}

	cfgType := reflect.TypeOf(models.ModelConfig{})
	snapType := reflect.TypeOf(ConfigSnapshot{})
	for i := 0; i < snapType.NumField(); i++ {
		f := snapType.Field(i)
		if notFromConfig[f.Name] {
			continue
		}
		cf, ok := cfgType.FieldByName(f.Name)
		if !ok {
			t.Errorf("ConfigSnapshot.%s has no ModelConfig field of the same name; copy it in SnapshotFromConfig and list it here", f.Name)
			continue
		}
		if cf.Type != f.Type {
			t.Errorf("ConfigSnapshot.%s is %s but ModelConfig.%s is %s", f.Name, f.Type, f.Name, cf.Type)
			continue
		}
		var cfg models.ModelConfig
		v := reflect.ValueOf(&cfg).Elem().FieldByName(f.Name)
		switch v.Kind() {
		case reflect.Bool:
			v.SetBool(true)
		case reflect.Int:
			v.SetInt(7)
		case reflect.String:
			v.SetString("x")
		default:
			t.Fatalf("%s: unhandled kind %s", f.Name, v.Kind())
		}
		snap := SnapshotFromConfig(cfg, "", false)
		if reflect.ValueOf(snap).FieldByName(f.Name).IsZero() {
			t.Errorf("SnapshotFromConfig does not copy %s", f.Name)
		}
	}

	snap := SnapshotFromConfig(models.ModelConfig{}, "Long context", true)
	if snap.ProfileName != "Long context" || !snap.ProfileEdited {
		t.Errorf("profile not recorded: %+v", snap)
	}
}

func TestMarkProfileEdited(t *testing.T) {
	base := ConfigSnapshot{ContextSize: 8192, ProfileName: "P"}

	if got := markProfileEdited(base, base); got.ProfileEdited {
		t.Error("an unchanged config is marked edited")
	}
	changed := base
	changed.UBatchSize = 1024
	if got := markProfileEdited(changed, base); !got.ProfileEdited {
		t.Error("an overridden config is not marked edited")
	}
	already := base
	already.ProfileEdited = true
	if got := markProfileEdited(already, already); !got.ProfileEdited {
		t.Error("a config edited since its profile lost the mark")
	}
}

// A job that overrides a setting did not run the profile exactly, and the
// run says so; a plain job over an unedited profile did.
func TestRunRecordsWhetherItRanTheProfile(t *testing.T) {
	saved := ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8, ProfileName: "Fast"}

	router := newFakeRouter(t)
	env := &fakeEnv{routerURL: router.URL, saved: saved}
	job, store := runJob(t, oneCellJob(nil), env)
	if runs := store.RunsForJob(job.ID); len(runs) != 1 || runs[0].Config.ProfileName != "Fast" || runs[0].Config.ProfileEdited {
		t.Errorf("plain run: %+v, want profile Fast, not edited", runs)
	}

	router = newFakeRouter(t)
	env = &fakeEnv{routerURL: router.URL, saved: saved}
	ctx := 16384
	job, store = runJob(t, oneCellJob(&ConfigOverrides{ContextSize: &ctx}), env)
	if runs := store.RunsForJob(job.ID); len(runs) != 1 || !runs[0].Config.ProfileEdited {
		t.Errorf("overridden run: %+v, want profile marked edited", runs)
	}
}

func TestCompareShowsProfileColumnOnlyWhenItVaries(t *testing.T) {
	a := BenchmarkRun{Config: ConfigSnapshot{ProfileName: "A"}}
	b := BenchmarkRun{Config: ConfigSnapshot{ProfileName: "A", ProfileEdited: true}}
	varies, _ := compareColumnVariance([]BenchmarkRun{a, b})
	if !varies["profile"] {
		t.Error("profile column hidden although one run was edited")
	}
	varies, _ = compareColumnVariance([]BenchmarkRun{a, a})
	if varies["profile"] {
		t.Error("profile column shown although every run shares it")
	}
	if got := ProfileCellText(b.Config); got != "A (edited)" {
		t.Errorf("ProfileCellText = %q", got)
	}
}
