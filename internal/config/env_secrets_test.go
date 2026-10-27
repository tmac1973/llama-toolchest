package config

import "testing"

func TestSecretEnvName(t *testing.T) {
	for _, n := range []string{"HF_TOKEN", "OPENAI_API_KEY", "my_secret", "DB_PASSWORD", "AWS_CREDENTIALS", "BASIC_AUTH"} {
		if !SecretEnvName(n) {
			t.Errorf("%s not treated as a secret", n)
		}
	}
	for _, n := range []string{"HSA_OVERRIDE_GFX_VERSION", "ROCBLAS_USE_HIPBLASLT", "FOO"} {
		if SecretEnvName(n) {
			t.Errorf("%s treated as a secret", n)
		}
	}
}

// The page shows secret values masked, and saving the masked text keeps
// the stored values; other lines, comments and blank lines are untouched.
func TestMaskAndUnmaskEnvText(t *testing.T) {
	stored := "# tokens\nHF_TOKEN=hf_real\nFOO=bar\n\nEMPTY_KEY="
	shown := MaskEnvText(stored)
	if want := "# tokens\nHF_TOKEN=" + EnvSecretMask + "\nFOO=bar\n\nEMPTY_KEY="; shown != want {
		t.Fatalf("masked = %q, want %q", shown, want)
	}
	if got := UnmaskEnvText(shown, stored); got != stored {
		t.Errorf("unmasked = %q, want the stored text %q", got, stored)
	}
	// A changed value is saved; a masked line with nothing stored is dropped.
	got := UnmaskEnvText("HF_TOKEN=hf_new\nNEW_TOKEN="+EnvSecretMask, stored)
	if got != "HF_TOKEN=hf_new" {
		t.Errorf("unmasked = %q, want HF_TOKEN=hf_new", got)
	}
}

func TestStripAndKeepSecretEnv(t *testing.T) {
	if got := StripSecretEnv("FOO=bar\nHF_TOKEN=hf_real\nAPI_KEY=k"); got != "FOO=bar" {
		t.Errorf("stripped = %q, want FOO=bar", got)
	}
	// A restore keeps the target's secrets the backup does not set, and
	// takes the backup's own value when it has one.
	got := KeepSecretEnv("FOO=new\nAPI_KEY=from_backup", "FOO=old\nHF_TOKEN=hf_target\nAPI_KEY=target")
	if want := "FOO=new\nAPI_KEY=from_backup\nHF_TOKEN=hf_target"; got != want {
		t.Errorf("kept = %q, want %q", got, want)
	}
}
