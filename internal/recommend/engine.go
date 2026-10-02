package recommend

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

const (
	// staleAfter is when a pool is marked old. It is still served; only
	// Refresh rebuilds it, and nothing runs in the background.
	staleAfter = 6 * time.Hour
	// buildTimeout caps a whole build. Finalists not checked by then are
	// listed as unverified.
	buildTimeout = 2 * time.Minute
	// verifyWorkers bounds the finalists checked at once.
	verifyWorkers = 8
)

// Engine builds and keeps the recommendation pool.
type Engine struct {
	// Hub returns the HuggingFace access to use, looked up on every build
	// so a token saved in Settings applies without a restart. Nil means
	// there is none.
	Hub func() Hub
	// Dir holds cached file listings; "" keeps nothing on disk.
	Dir string
	// Now is the clock, for tests; nil is time.Now.
	Now func() time.Time

	mu    sync.Mutex
	pool  *Pool
	build *buildCall
}

type buildCall struct {
	key  string
	done chan struct{}
	pool *Pool
}

func (e *Engine) now() time.Time {
	if e.Now != nil {
		return e.Now()
	}
	return time.Now()
}

// Pool is one build's result.
type Pool struct {
	Key     string
	Profile Profile
	Built   time.Time
	// Unavailable is why there is no list at all, as a sentence; "" when
	// the build ran.
	Unavailable string
	// Groups are the verified models; Unverified those that could not be
	// fully checked.
	Groups     []*Group
	Unverified []Unverified
	orders     map[Intent]map[models.ContextClass][]*Group
}

// Order is the models with a pick at class, in one order.
func (p *Pool) Order(intent Intent, class models.ContextClass) []*Group {
	if p == nil || p.orders == nil {
		return nil
	}
	return p.orders[intent][class]
}

// Hidden is how many verified models have no pick at class: they run
// only with a shorter context, or were trained for less.
func (p *Pool) Hidden(class models.ContextClass) int {
	n := 0
	for _, g := range p.Groups {
		if g.Picks[class] == nil {
			n++
		}
	}
	return n
}

// Stale reports whether the pool is older than staleAfter.
func (p *Pool) Stale(now time.Time) bool {
	return now.Sub(p.Built) > staleAfter
}

// Get returns the pool for profile, building it when there is none for
// this profile yet or when refresh is set. Concurrent callers share one
// build. The build runs on its own deadline, so a caller that gives up
// does not cancel it for the others; Get then returns nil.
func (e *Engine) Get(ctx context.Context, p Profile, refresh bool) *Pool {
	key := p.Key()
	e.mu.Lock()
	if !refresh && e.pool != nil && e.pool.Key == key {
		pool := e.pool
		e.mu.Unlock()
		return pool
	}
	call := e.build
	if call == nil || call.key != key {
		call = &buildCall{key: key, done: make(chan struct{})}
		e.build = call
		go func() {
			pool := e.run(p)
			e.mu.Lock()
			e.pool = pool
			if e.build == call {
				e.build = nil
			}
			e.mu.Unlock()
			call.pool = pool
			close(call.done)
		}()
	}
	e.mu.Unlock()

	select {
	case <-call.done:
		return call.pool
	case <-ctx.Done():
		return nil
	}
}

// Current is the pool last built, or nil.
func (e *Engine) Current() *Pool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.pool
}

// Suggested lists the context classes at which the current pool picks
// file from repo.
func (e *Engine) Suggested(repo, file string) []models.ContextClass {
	pool := e.Current()
	if pool == nil {
		return nil
	}
	var out []models.ContextClass
	for _, g := range pool.Groups {
		if g.Repo().ID != repo {
			continue
		}
		for _, class := range Classes {
			if p := g.Picks[class]; p != nil && p.File == file {
				out = append(out, class)
			}
		}
	}
	return out
}

// run builds a pool from scratch.
func (e *Engine) run(p Profile) *Pool {
	start := e.now()
	pool := &Pool{Key: p.Key(), Profile: p, Built: start}
	if len(models.PlanCards(p.Hardware)) == 0 {
		pool.Unavailable = "No GPU was detected, so there is nothing to plan for."
		return pool
	}
	var hub Hub
	if e.Hub != nil {
		hub = e.Hub()
	}
	if hub == nil {
		pool.Unavailable = "Hugging Face is not available."
		return pool
	}

	ctx, cancel := context.WithTimeout(context.Background(), buildTimeout)
	defer cancel()

	listed, err := fetchCandidates(ctx, hub)
	if err != nil {
		pool.Unavailable = "Could not reach Hugging Face (" + err.Error() + ")."
		return pool
	}
	var keep = listed[:0:0]
	for _, r := range listed {
		if judgeCandidate(r, p) != candidateDrop {
			keep = append(keep, r)
		}
	}
	finalists := chooseFinalists(groupRepos(keep, p), p.Hardware)

	verdicts := make([]verdict, len(finalists))
	sem := make(chan struct{}, verifyWorkers)
	var wg sync.WaitGroup
	for i, g := range finalists {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			if ctx.Err() != nil {
				verdicts[i] = verdict{unverified: reasonTimeout}
				return
			}
			verdicts[i] = e.verify(ctx, hub, g, p)
		}()
	}
	wg.Wait()

	dropped := 0
	for i, g := range finalists {
		switch v := verdicts[i]; {
		case v.dropped:
			dropped++
		case v.unverified != "":
			pool.Unverified = append(pool.Unverified, Unverified{Repo: g.Repo().ID, Gated: g.Gated, Reason: v.unverified})
		default:
			pool.Groups = append(pool.Groups, g)
		}
	}
	pool.orders = orderAll(pool.Groups)
	slog.Info("recommendations built",
		"candidates", len(listed), "finalists", len(finalists), "verified", len(pool.Groups),
		"unverified", len(pool.Unverified), "dropped", dropped, "took", e.now().Sub(start).Round(time.Millisecond))
	return pool
}
