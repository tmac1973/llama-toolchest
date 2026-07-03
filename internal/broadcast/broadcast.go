// Package broadcast provides the mutex-protected fan-out primitive shared
// by the build-log, router-log, and download-progress streams: subscribers
// are channels, recent values are retained and replayed to new subscribers,
// and sends never block (a subscriber that falls behind misses values
// rather than stalling the producer).
package broadcast

import "sync"

// Broadcaster fans values out to subscriber channels.
type Broadcaster[T any] struct {
	mu       sync.Mutex
	history  []T
	histSize int
	chanCap  int
	subs     map[chan T]struct{}
}

// New creates a Broadcaster retaining up to histSize values as replayable
// history (0 = no history). Subscriber channels are buffered to chanCap.
func New[T any](histSize, chanCap int) *Broadcaster[T] {
	return &Broadcaster[T]{
		histSize: histSize,
		chanCap:  chanCap,
		subs:     make(map[chan T]struct{}),
	}
}

// Subscribe registers a new subscriber channel, replaying retained history
// into it first.
func (b *Broadcaster[T]) Subscribe() chan T {
	b.mu.Lock()
	defer b.mu.Unlock()
	ch := make(chan T, b.chanCap)
	for _, v := range b.history {
		select {
		case ch <- v:
		default:
		}
	}
	b.subs[ch] = struct{}{}
	return ch
}

// Unsubscribe removes a subscriber. The channel is not closed.
func (b *Broadcaster[T]) Unsubscribe(ch chan T) {
	b.mu.Lock()
	delete(b.subs, ch)
	b.mu.Unlock()
}

// Broadcast retains v in history (dropping the oldest value when full) and
// sends it to all subscribers without blocking.
func (b *Broadcaster[T]) Broadcast(v T) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.histSize > 0 {
		if len(b.history) >= b.histSize {
			b.history = b.history[1:]
		}
		b.history = append(b.history, v)
	}
	for ch := range b.subs {
		select {
		case ch <- v:
		default:
		}
	}
}

// Last returns the most recently broadcast value, if any is retained.
func (b *Broadcaster[T]) Last() (T, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.history) == 0 {
		var zero T
		return zero, false
	}
	return b.history[len(b.history)-1], true
}

// ClearHistory discards retained history so it won't replay to new
// subscribers. Existing subscribers are unaffected.
func (b *Broadcaster[T]) ClearHistory() {
	b.mu.Lock()
	b.history = nil
	b.mu.Unlock()
}

// CloseSubscribers closes and drops all subscriber channels, signalling
// end-of-stream. History is kept so later subscribers still get a replay.
func (b *Broadcaster[T]) CloseSubscribers() {
	b.mu.Lock()
	defer b.mu.Unlock()
	for ch := range b.subs {
		close(ch)
	}
	b.subs = make(map[chan T]struct{})
}
