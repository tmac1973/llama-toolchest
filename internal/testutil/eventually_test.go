package testutil

import (
	"sync/atomic"
	"testing"
)

func TestEventuallyReturnsOnceTheConditionHolds(t *testing.T) {
	var n atomic.Int32
	Eventually(t, func() bool { return n.Add(1) >= 3 }, "never reached 3")
	if n.Load() != 3 {
		t.Errorf("checked %d times, want 3", n.Load())
	}
}
