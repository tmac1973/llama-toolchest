package generation

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
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
