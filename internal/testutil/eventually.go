// Package testutil holds helpers that tests in several packages share.
// Only test files import it.
package testutil

import (
	"fmt"
	"testing"
	"time"
)

// EventuallyTimeout is how long Eventually waits. It is generous, since a
// passing test returns as soon as the condition holds, and a loaded CI
// machine can be slow.
const EventuallyTimeout = 60 * time.Second

// Eventually checks cond every few milliseconds until it is true, and
// fails the test with the message when it is still false after
// EventuallyTimeout. Call it only from the test's own goroutine.
func Eventually(t testing.TB, cond func() bool, format string, args ...any) {
	t.Helper()
	EventuallyMsg(t, cond, func() string { return fmt.Sprintf(format, args...) })
}

// EventuallyMsg is Eventually with the message built when the wait fails,
// so it can show the last state seen.
func EventuallyMsg(t testing.TB, cond func() bool, msg func() string) {
	t.Helper()
	deadline := time.Now().Add(EventuallyTimeout)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal("timed out: " + msg())
		}
		time.Sleep(5 * time.Millisecond)
	}
}
