package generation

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// Jobs on one GPU run strictly one after another, in order.
func TestQueueOneJobPerGPU(t *testing.T) {
	q := NewQueue()
	var mu sync.Mutex
	var order []int64
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		tk, err := q.Submit(0, "img", KindSD)
		if err != nil {
			t.Fatal(err)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := tk.Wait(context.Background()); err != nil {
				t.Error(err)
				return
			}
			if q.Load(0) == 0 || q.Snapshot(0).Running.ID != tk.Info().ID {
				t.Errorf("job %d runs but is not the GPU's running job", tk.Info().ID)
			}
			mu.Lock()
			order = append(order, tk.Info().ID)
			mu.Unlock()
			time.Sleep(20 * time.Millisecond)
			tk.Done()
		}()
	}
	wg.Wait()
	h := q.History(0)
	if len(h) != 3 {
		t.Fatalf("history has %d jobs, want 3", len(h))
	}
	for i := 1; i < len(h); i++ {
		if h[i].Started.Before(h[i-1].Finished) {
			t.Errorf("job %d started before job %d finished", h[i].ID, h[i-1].ID)
		}
	}
	if order[0] != 1 || order[1] != 2 || order[2] != 3 {
		t.Errorf("run order %v, want 1 2 3", order)
	}
}

// Jobs on different GPUs run at the same time.
func TestQueueGPUsIndependent(t *testing.T) {
	q := NewQueue()
	a, _ := q.Submit(0, "a", KindSD)
	b, _ := q.Submit(1, "b", KindAce)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if a.Wait(ctx) != nil || b.Wait(ctx) != nil {
		t.Fatal("a job on a free GPU had to wait")
	}
	a.Done()
	b.Done()
}

func TestQueueFull(t *testing.T) {
	q := NewQueue()
	if _, err := q.Submit(0, "x", KindSD); err != nil { // running
		t.Fatal(err)
	}
	for i := 0; i < MaxWaitingPerGPU; i++ {
		if _, err := q.Submit(0, "x", KindSD); err != nil {
			t.Fatalf("waiting job %d: %v", i+1, err)
		}
	}
	if _, err := q.Submit(0, "x", KindSD); !errors.Is(err, ErrQueueFull) {
		t.Errorf("job %d: %v, want ErrQueueFull", MaxWaitingPerGPU+2, err)
	}
	if _, err := q.Submit(1, "x", KindSD); err != nil {
		t.Errorf("another GPU: %v", err)
	}
}

// A waiting job whose client goes away leaves the queue, and the next one
// moves up.
func TestQueueCancelWaiting(t *testing.T) {
	q := NewQueue()
	run, _ := q.Submit(0, "x", KindSD)
	gone, _ := q.Submit(0, "x", KindSD)
	next, _ := q.Submit(0, "x", KindSD)
	if gone.Position() != 1 || next.Position() != 2 {
		t.Fatalf("positions %d %d, want 1 2", gone.Position(), next.Position())
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := gone.Wait(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled wait: %v", err)
	}
	if next.Position() != 1 {
		t.Errorf("next job's position %d, want 1", next.Position())
	}
	run.Done()
	wctx, wcancel := context.WithTimeout(context.Background(), time.Second)
	defer wcancel()
	if err := next.Wait(wctx); err != nil {
		t.Fatalf("next job never started: %v", err)
	}
	if h := q.History(0); len(h) != 1 {
		t.Errorf("history %v: the cancelled job should not be in it", h)
	}
	next.Done()
	next.Done() // a second Done is a no-op
	if q.Load(0) != 0 {
		t.Error("queue not empty")
	}
}

// An LLM measurement's hold waits for the running image job, keeps new
// jobs waiting while it has the GPU, and lets them run when released.
func TestQueueHold(t *testing.T) {
	q := NewQueue()
	img, _ := q.Submit(0, "img", KindSD)
	if err := img.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}

	held := make(chan func())
	go func() {
		release, err := q.Hold(context.Background(), []int{0}, "a benchmark")
		if err != nil {
			t.Error(err)
		}
		held <- release
	}()
	select {
	case <-held:
		t.Fatal("hold taken while an image job runs")
	case <-time.After(30 * time.Millisecond):
	}

	// A job arriving while the hold waits queues behind it, not ahead.
	song, _ := q.Submit(0, "ace", KindAce)
	img.Done()
	var release func()
	select {
	case release = <-held:
	case <-time.After(time.Second):
		t.Fatal("hold not taken after the image job finished")
	}
	if snap := q.Snapshot(0); snap.HeldBy != "a benchmark" || snap.Running != nil || len(snap.Waiting) != 1 {
		t.Errorf("while held: %+v, want held by the benchmark with the song waiting", snap)
	}
	if q.Load(0) != 2 {
		t.Errorf("load while held = %d, want 2 (the hold and the waiting song)", q.Load(0))
	}

	release()
	release() // a second call does nothing
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := song.Wait(ctx); err != nil {
		t.Fatalf("song did not start after the hold ended: %v", err)
	}
	song.Done()

	h := q.History(0)
	if len(h) != 3 || h[1].Kind != KindLLM || h[1].InstanceID != "a benchmark" {
		t.Fatalf("history = %+v, want image, hold, song", h)
	}
	for i := 1; i < len(h); i++ {
		if h[i].Started.Before(h[i-1].Finished) {
			t.Errorf("%s started before %s finished", h[i].InstanceID, h[i-1].InstanceID)
		}
	}
}

// A hold that gives up waiting leaves the GPU to the jobs queued behind it.
func TestQueueHoldCancelled(t *testing.T) {
	q := NewQueue()
	img, _ := q.Submit(0, "img", KindSD)
	img.Wait(context.Background())
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error)
	go func() {
		_, err := q.Hold(ctx, []int{0}, "a benchmark")
		done <- err
	}()
	// Submit once the hold has claimed the GPU, so the job queues behind it.
	testutil.Eventually(t, func() bool {
		q.mu.Lock()
		defer q.mu.Unlock()
		return q.gpu(0).claims > 0
	}, "the hold never claimed the GPU")
	waiting, _ := q.Submit(0, "img", KindSD)
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled hold returned %v", err)
	}
	img.Done()
	wctx, wcancel := context.WithTimeout(context.Background(), time.Second)
	defer wcancel()
	if err := waiting.Wait(wctx); err != nil {
		t.Errorf("job behind a cancelled hold did not start: %v", err)
	}
}

// A kept GPU stays closed to waiting jobs between one hold and the next,
// and a hold on it is not kept waiting by the keep itself.
func TestQueueKeep(t *testing.T) {
	q := NewQueue()
	first, err := q.Hold(context.Background(), []int{1, 2}, "autotune")
	if err != nil {
		t.Fatal(err)
	}
	keep := q.Keep([]int{2, 1, 2}, "autotune")
	song, _ := q.Submit(2, "ace", KindAce)
	first()

	// The gap between two cells: the song must not start.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if err := song.Wait(ctx); err == nil {
		t.Fatal("song started while the GPU was kept")
	}
	song, _ = q.Submit(2, "ace", KindAce)
	if snap := q.Snapshot(2); snap.HeldBy != "autotune" || snap.Running != nil || len(snap.Waiting) != 1 {
		t.Errorf("between holds: %+v, want kept by autotune with the song waiting", snap)
	}
	if q.Load(2) != 2 {
		t.Errorf("load while kept = %d, want 2 (the keep and the waiting song)", q.Load(2))
	}

	hctx, hcancel := context.WithTimeout(context.Background(), time.Second)
	defer hcancel()
	second, err := q.Hold(hctx, []int{1, 2}, "autotune")
	if err != nil {
		t.Fatalf("hold on a kept GPU did not go through: %v", err)
	}
	second()
	if q.Snapshot(1).HeldBy != "autotune" {
		t.Error("GPU 1 not kept after the second hold ended")
	}

	keep()
	keep() // a second call does nothing
	wctx, wcancel := context.WithTimeout(context.Background(), time.Second)
	defer wcancel()
	if err := song.Wait(wctx); err != nil {
		t.Fatalf("song did not start after the keep ended: %v", err)
	}
	song.Done()
	if snap := q.Snapshot(1); snap.HeldBy != "" {
		t.Errorf("GPU 1 still shows %q after the keep ended", snap.HeldBy)
	}
}

// A keep taken while a job runs lets it finish, then holds the queue.
func TestQueueKeepRunningJob(t *testing.T) {
	q := NewQueue()
	img, _ := q.Submit(0, "img", KindSD)
	if err := img.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	keep := q.Keep([]int{0}, "a benchmark")
	defer keep()
	next, _ := q.Submit(0, "img", KindSD)
	img.Done()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if err := next.Wait(ctx); err == nil {
		t.Fatal("a waiting job started while the GPU was kept")
	}
}
