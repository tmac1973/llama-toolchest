package broadcast

import (
	"sync"
	"testing"
)

func TestHistoryReplayAndRing(t *testing.T) {
	b := New[int](3, 8)
	for i := 1; i <= 5; i++ {
		b.Broadcast(i)
	}
	ch := b.Subscribe()
	// Ring size 3 → only 3, 4, 5 retained.
	for _, want := range []int{3, 4, 5} {
		if got := <-ch; got != want {
			t.Fatalf("replay = %d, want %d", got, want)
		}
	}
	if last, ok := b.Last(); !ok || last != 5 {
		t.Fatalf("Last() = %d, %v; want 5, true", last, ok)
	}
}

func TestSubscribeReceivesNewValues(t *testing.T) {
	b := New[string](0, 4)
	ch := b.Subscribe()
	b.Broadcast("a")
	if got := <-ch; got != "a" {
		t.Fatalf("got %q, want %q", got, "a")
	}
	b.Unsubscribe(ch)
	b.Broadcast("b")
	select {
	case v := <-ch:
		t.Fatalf("received %q after unsubscribe", v)
	default:
	}
}

func TestNonBlockingSendDropsWhenFull(t *testing.T) {
	b := New[int](0, 1)
	ch := b.Subscribe()
	b.Broadcast(1)
	b.Broadcast(2) // dropped: subscriber buffer full
	if got := <-ch; got != 1 {
		t.Fatalf("got %d, want 1", got)
	}
	select {
	case v := <-ch:
		t.Fatalf("expected drop, received %d", v)
	default:
	}
}

func TestCloseSubscribersKeepsHistory(t *testing.T) {
	b := New[int](2, 4)
	ch := b.Subscribe()
	b.Broadcast(7)
	b.CloseSubscribers()
	// Drain the value, then confirm the channel is closed.
	if got := <-ch; got != 7 {
		t.Fatalf("got %d, want 7", got)
	}
	if _, open := <-ch; open {
		t.Fatal("channel still open after CloseSubscribers")
	}
	// History survives for later subscribers.
	ch2 := b.Subscribe()
	if got := <-ch2; got != 7 {
		t.Fatalf("replay after close = %d, want 7", got)
	}
}

func TestClearHistory(t *testing.T) {
	b := New[int](4, 4)
	b.Broadcast(1)
	b.ClearHistory()
	ch := b.Subscribe()
	select {
	case v := <-ch:
		t.Fatalf("unexpected replay %d after ClearHistory", v)
	default:
	}
}

func TestConcurrentUse(t *testing.T) {
	b := New[int](16, 16)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				b.Broadcast(j)
			}
		}()
		go func() {
			defer wg.Done()
			ch := b.Subscribe()
			for j := 0; j < 50; j++ {
				select {
				case <-ch:
				default:
				}
			}
			b.Unsubscribe(ch)
		}()
	}
	wg.Wait()
}
