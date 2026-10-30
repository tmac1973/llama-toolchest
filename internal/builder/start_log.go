package builder

import (
	"context"
	"regexp"
	"sort"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"
)

// A build has no ID until its source is fetched and its ref resolved,
// which can take from seconds to minutes (a first clone, or a fetch of
// every llama.cpp tag). The page names the start with a token instead, so
// it can show those git lines while they happen: it sends the token with
// the start request and reads the lines from SubscribeStart(token).

type startTokenKey struct{}

// startTokenPattern is what a start token may look like: the page makes
// it from random letters and digits.
var startTokenPattern = regexp.MustCompile(`^[a-z0-9]{8,64}$`)

// ValidStartToken reports whether token can name a build start.
func ValidStartToken(token string) bool { return startTokenPattern.MatchString(token) }

// WithStartToken returns ctx carrying the token of the build start that
// BuildProject runs, so its fetch and checkout lines go to that token's
// stream.
func WithStartToken(ctx context.Context, token string) context.Context {
	if !ValidStartToken(token) {
		return ctx
	}
	return context.WithValue(ctx, startTokenKey{}, token)
}

func startTokenFrom(ctx context.Context) string {
	t, _ := ctx.Value(startTokenKey{}).(string)
	return t
}

// startLogTTL is how long a start stream is kept. The page reads it only
// until the start request returns; the time limit only frees streams of
// starts that failed early or were never made.
const startLogTTL = 10 * time.Minute

// startLog is the stream of one build start: its lines, and whether it
// has ended, so a reader that comes late gets the lines and the end.
type startLog struct {
	bc    *broadcast.Broadcaster[string]
	ended bool
}

type startLogs struct {
	mu   sync.Mutex
	logs map[string]*startLog
}

// get returns the stream of token, made if needed. Whichever comes first,
// the page's reader or the start request, makes it.
func (s *startLogs) get(token string) *startLog {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.logs == nil {
		s.logs = map[string]*startLog{}
	}
	l, ok := s.logs[token]
	if !ok {
		l = &startLog{bc: broadcast.New[string](buildLogHistorySize, buildLogHistorySize)}
		s.logs[token] = l
		time.AfterFunc(startLogTTL, func() {
			s.mu.Lock()
			delete(s.logs, token)
			s.mu.Unlock()
		})
	}
	return l
}

// end closes the stream: current readers see the end, and later ones get
// the lines and then the end.
func (s *startLogs) end(l *startLog) {
	s.mu.Lock()
	l.ended = true
	s.mu.Unlock()
	l.bc.CloseSubscribers()
}

// SubscribeStart returns the fetch and checkout lines of the build start
// named by token, and the function to call when done reading. The channel
// is closed when the start has ended. ok is false for a malformed token.
func (b *Builder) SubscribeStart(token string) (ch chan string, done func(), ok bool) {
	if !ValidStartToken(token) {
		return nil, nil, false
	}
	l := b.starts.get(token)
	b.starts.mu.Lock()
	ended := l.ended
	b.starts.mu.Unlock()
	ch = l.bc.Subscribe()
	if ended {
		l.bc.Unsubscribe(ch)
		close(ch)
		return ch, func() {}, true
	}
	return ch, func() { l.bc.Unsubscribe(ch) }, true
}

// RunningBuilds returns the builds that are still running, newest first.
func (b *Builder) RunningBuilds() []BuildResult {
	b.mu.Lock()
	defer b.mu.Unlock()
	var out []BuildResult
	for _, br := range b.builds {
		if br.Status == BuildStatusBuilding {
			out = append(out, br)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].StartedAt.After(out[j].StartedAt) })
	return out
}
