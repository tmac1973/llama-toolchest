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

// History is how a caller reads everything still held. A subscription
// cannot: its replay stops at the channel's capacity.
func TestHistoryReturnsEverythingHeld(t *testing.T) {
	b := New[int](5, 2)
	for i := 1; i <= 7; i++ {
		b.Broadcast(i)
	}
	got := b.History()
	want := []int{3, 4, 5, 6, 7}
	if len(got) != len(want) {
		t.Fatalf("history = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("history = %v, want %v", got, want)
		}
	}

	// A copy: changing it must not change what the broadcaster holds.
	got[0] = 99
	if again := b.History(); again[0] != 3 {
		t.Errorf("changing the returned slice changed the held history: %v", again)
	}
}

// A new subscriber gets everything held, newest included, even when the
// history is longer than the channel's usual capacity. It used to get the
// oldest values and lose the newest: with 500 lines of router log held
// and room for 256, the Server Logs panel opened on the first 256.
func TestSubscribeReplaysTheNewestValues(t *testing.T) {
	b := New[int](500, 256)
	for i := 1; i <= 500; i++ {
		b.Broadcast(i)
	}
	ch := b.Subscribe()
	defer b.Unsubscribe(ch)

	var got []int
	for len(ch) > 0 {
		got = append(got, <-ch)
	}
	if len(got) != 500 || got[0] != 1 || got[len(got)-1] != 500 {
		t.Fatalf("replayed %d values from %v to %v, want all 500 from 1 to 500",
			len(got), first(got), last(got))
	}

	// Live values still arrive after the replay, up to the usual capacity.
	b.Broadcast(501)
	if v := <-ch; v != 501 {
		t.Errorf("live value = %d, want 501", v)
	}
}

func first(xs []int) any {
	if len(xs) == 0 {
		return "nothing"
	}
	return xs[0]
}

func last(xs []int) any {
	if len(xs) == 0 {
		return "nothing"
	}
	return xs[len(xs)-1]
}
