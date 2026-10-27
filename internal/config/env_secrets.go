package config

import "strings"

// EnvSecretMask stands in for the value of a secret-looking runtime
// environment line wherever the line is shown, since the pages that show it
// have no login. Saving a line with this value keeps the stored value.
const EnvSecretMask = "********"

// secretEnvWords mark an environment variable name as holding a secret.
var secretEnvWords = []string{"TOKEN", "SECRET", "PASSWORD", "PASSWD", "KEY", "CREDENTIAL", "AUTH"}

// SecretEnvName reports whether an environment variable's name suggests
// its value is a secret, such as HF_TOKEN or OPENAI_API_KEY.
func SecretEnvName(name string) bool {
	upper := strings.ToUpper(name)
	for _, w := range secretEnvWords {
		if strings.Contains(upper, w) {
			return true
		}
	}
	return false
}

// secretEnvLine reports the name of a KEY=VALUE line whose name looks
// secret, or "" for any other line (comments, blank and malformed lines
// included).
func secretEnvLine(line string) string {
	t := strings.TrimSpace(line)
	if t == "" || strings.HasPrefix(t, "#") {
		return ""
	}
	eq := strings.IndexByte(t, '=')
	if eq <= 0 || !validEnvName(t[:eq]) || !SecretEnvName(t[:eq]) {
		return ""
	}
	return t[:eq]
}

// MaskEnvText returns KEY=VALUE text with the value of each secret-looking
// line replaced by EnvSecretMask. Other lines are kept as they are.
func MaskEnvText(raw string) string {
	lines := strings.Split(raw, "\n")
	for i, line := range lines {
		if name := secretEnvLine(line); name != "" && strings.TrimSpace(line) != name+"=" {
			lines[i] = name + "=" + EnvSecretMask
		}
	}
	return strings.Join(lines, "\n")
}

// UnmaskEnvText returns submitted text, from a form that showed masked
// text, with each NAME=EnvSecretMask line given back its value from
// current. A masked line with no value in current is dropped.
func UnmaskEnvText(submitted, current string) string {
	stored := map[string]string{}
	for _, line := range strings.Split(current, "\n") {
		if name := secretEnvLine(line); name != "" {
			stored[name] = strings.TrimSpace(line)
		}
	}
	var out []string
	for _, line := range strings.Split(submitted, "\n") {
		name := secretEnvLine(line)
		if name == "" || strings.TrimSpace(line) != name+"="+EnvSecretMask {
			out = append(out, line)
			continue
		}
		if prev, ok := stored[name]; ok {
			out = append(out, prev)
		}
	}
	return strings.Join(out, "\n")
}

// StripSecretEnv returns KEY=VALUE text without its secret-looking lines,
// for a backup made without secrets.
func StripSecretEnv(raw string) string {
	var out []string
	for _, line := range strings.Split(raw, "\n") {
		if secretEnvLine(line) == "" {
			out = append(out, line)
		}
	}
	return strings.TrimRight(strings.Join(out, "\n"), "\n")
}

// KeepSecretEnv returns restored KEY=VALUE text with the secret-looking
// lines of current that restored does not set added at the end, so a
// restore from a backup made without secrets does not drop them.
func KeepSecretEnv(restored, current string) string {
	have := map[string]bool{}
	for _, line := range strings.Split(restored, "\n") {
		if name := secretEnvLine(line); name != "" {
			have[name] = true
		}
	}
	out := strings.TrimRight(restored, "\n")
	for _, line := range strings.Split(current, "\n") {
		if name := secretEnvLine(line); name != "" && !have[name] {
			out += "\n" + strings.TrimSpace(line)
			have[name] = true
		}
	}
	return strings.TrimLeft(out, "\n")
}
