package api

import (
	"context"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// GPUs a job's cells claim stay kept for it until ReleaseGPUs, and a
// claim for another owner gives back anything still kept for the last.
func TestJobEnvKeepsGPUsForOwner(t *testing.T) {
	q := generation.NewQueue()
	e := &jobEnv{s: &Server{genQueue: q}}

	e.keepGPUs("autotune:a", []int{1, 2}, "Autotune: m")
	e.keepGPUs("autotune:a", []int{2}, "Autotune: m") // a later cell: nothing new
	song, _ := q.Submit(2, "ace-gpu2", generation.KindAce)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if err := song.Wait(ctx); err == nil {
		t.Fatal("a music job started on a kept GPU")
	}
	song, _ = q.Submit(2, "ace-gpu2", generation.KindAce)

	e.ReleaseGPUs("job:other") // not the owner: nothing changes
	if q.Snapshot(2).HeldBy == "" {
		t.Fatal("GPU 2 given back by another owner's release")
	}

	// The next job takes GPU 0; what the last run kept is given back.
	e.keepGPUs("job:b", []int{0}, "a benchmark")
	wctx, wcancel := context.WithTimeout(context.Background(), time.Second)
	defer wcancel()
	if err := song.Wait(wctx); err != nil {
		t.Fatalf("music job did not start once another job claimed GPUs: %v", err)
	}
	song.Done()
	if q.Snapshot(1).HeldBy != "" || q.Snapshot(0).HeldBy != "a benchmark" {
		t.Errorf("GPU 1 %q, GPU 0 %q; want GPU 1 free and GPU 0 kept", q.Snapshot(1).HeldBy, q.Snapshot(0).HeldBy)
	}
	e.ReleaseGPUs("job:b")
	if q.Snapshot(0).HeldBy != "" {
		t.Error("GPU 0 still kept after its job released it")
	}
}
