package readonly

import (
	"errors"
	"fmt"
	"io/fs"
	"strings"
	"testing"
)

var errTest = errors.New("test store is read-only")

func TestCheck(t *testing.T) {
	if err := Check("", errTest); err != nil {
		t.Errorf("Check with no reason = %v, want nil", err)
	}
	err := Check("models.json could not be parsed", errTest)
	if !errors.Is(err, errTest) {
		t.Errorf("errors.Is(%v, sentinel) = false", err)
	}
	if err.Error() != "models.json could not be parsed" {
		t.Errorf("Error() = %q, want the reason", err.Error())
	}
}

// A missing file is a first start; any other read error is a reason.
func TestReadReason(t *testing.T) {
	if r := ReadReason("x.json", "X", fmt.Errorf("open: %w", fs.ErrNotExist)); r != "" {
		t.Errorf("missing file: reason %q, want none", r)
	}
	if r := ReadReason("x.json", "X", fs.ErrPermission); !strings.Contains(r, "x.json could not be read") {
		t.Errorf("unreadable file: reason %q", r)
	}
}
