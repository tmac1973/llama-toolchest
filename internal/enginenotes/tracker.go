package enginenotes

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/serverlog"
)

// OutcomeKind says what happened. A string, because it is written to JSON
// and read in templates, where a number would mean nothing.
type OutcomeKind string

const (
	// KindLoad is a model instance that finished loading.
	KindLoad OutcomeKind = "load"
	// KindLoadFailed is an instance that exited with an error status
	// before it finished loading, without being asked to stop.
	KindLoadFailed OutcomeKind = "load_failed"
	// KindCrash is an instance that had loaded and then exited with an
	// error status, without being asked to stop.
	KindCrash OutcomeKind = "crash"
	// KindUnload is an instance that ended for any other reason: an
	// unload, an eviction, a stop that had to be forced, or the router
	// ending under it.
	KindUnload OutcomeKind = "unload"
	// KindRouterStop is the router stopped on request.
	KindRouterStop OutcomeKind = "router_stop"
	// KindRouterExit is the router exiting when nobody asked it to.
	KindRouterExit OutcomeKind = "router_exit"
)

// tailLines is how many of an instance's own lines are kept, so a crash
// can show what the instance wrote just before it died.
const tailLines = 30

// Outcome is one thing the log showed happening.
type Outcome struct {
	Kind OutcomeKind
	Time time.Time
	// RouterName is the router's name for the model (its name=), and
	// Port the instance's port. Both are empty for the router's own
	// outcomes.
	RouterName string
	Port       string
	// ExitStatus is the instance's exit status, on the kinds that end
	// an instance.
	ExitStatus int
	// LoadSeconds is spawn to loaded, set on KindLoad.
	LoadSeconds float64
	// LoadedSeconds is loaded to exit, set on KindCrash and KindUnload
	// for an instance that had loaded.
	LoadedSeconds float64
	// ErrorLines and Tail are set on KindCrash and KindLoadFailed: the
	// lines that say what went wrong, and the instance's last lines as
	// written.
	ErrorLines []string
	Tail       []string
	// RouterError is why the router exited, on KindRouterExit.
	RouterError string
}

// OpenInstance is a model instance that has loaded and not yet ended.
type OpenInstance struct {
	RouterName string
	Port       string
	ReadySince time.Time
}

type instance struct {
	name      string
	port      string
	spawnedAt time.Time
	readyAt   time.Time
	tail      []string
}

// The router's own lines about its instances (llama.cpp
// tools/server/server-models.cpp). Names are registry ids, which hold no
// spaces, but the expressions read to the end of the line anyway, as
// serverlog does for the spawn line.
var (
	stoppingRE = regexp.MustCompile(`stopping model instance name=(.+?)\s*$`)
	exitedRE   = regexp.MustCompile(`instance name=(.+) exited with status (-?\d+)\s*$`)
)

// The lines the toolchest's supervisor adds to the router's log when the
// router ends (process.Supervisor, label "Router").
const (
	routerStopped      = "==> Router stopped"
	routerExitedError  = "==> Router exited with error: "
	routerExitedNormal = "==> Router exited normally"
)

// Tracker turns the router's log, line by line, into outcomes. It is not
// safe for use from more than one goroutine: one reader feeds it.
type Tracker struct {
	byPort map[string]*instance
	// stopping holds the names the router has been told to stop, so
	// their exit is an unload whatever status it reports.
	stopping map[string]bool
}

// NewTracker returns a tracker that has seen nothing.
func NewTracker() *Tracker {
	return &Tracker{byPort: map[string]*instance{}, stopping: map[string]bool{}}
}

// Add reads one log line, seen at now, and returns what it showed
// happening: usually nothing.
func (t *Tracker) Add(line string, now time.Time) []Outcome {
	line = strings.TrimPrefix(line, "data: ")

	if name, port, ok := serverlog.ParseSpawn(line); ok {
		t.byPort[port] = &instance{name: name, port: port, spawnedAt: now}
		// A new instance of a name starts with no stop pending.
		delete(t.stopping, name)
		return nil
	}

	// Before the prefix test: the router's ready line starts with its own
	// timestamp, not with the instance's port.
	load := t.ready(line, now)

	port, fromInstance := serverlog.InstancePort(line)
	if in := t.byPort[port]; fromInstance && in != nil {
		in.tail = append(in.tail, line)
		if len(in.tail) > tailLines {
			in.tail = in.tail[len(in.tail)-tailLines:]
		}
	}
	if load != nil {
		return []Outcome{*load}
	}
	if fromInstance {
		return nil
	}

	if m := stoppingRE.FindStringSubmatch(line); m != nil {
		t.stopping[m[1]] = true
		return nil
	}

	if m := exitedRE.FindStringSubmatch(line); m != nil {
		status, _ := strconv.Atoi(m[2])
		return []Outcome{t.exited(m[1], status, now)}
	}

	switch {
	case line == routerStopped:
		return t.routerEnded(Outcome{Kind: KindRouterStop, Time: now})
	case strings.HasPrefix(line, routerExitedError):
		return t.routerEnded(Outcome{Kind: KindRouterExit, Time: now, RouterError: strings.TrimPrefix(line, routerExitedError)})
	case line == routerExitedNormal:
		// The supervisor prints "stopped" after a stop it was asked for,
		// so a router that exits on its own reaches here.
		return t.routerEnded(Outcome{Kind: KindRouterExit, Time: now})
	}
	return nil
}

// ready returns a load outcome when the line says an instance has
// finished loading. Either of two lines says so: the instance's own
// "model loaded", which is there at the default verbosity, and the
// router's "ready" state line, which needs verbosity 5. The first one
// seen counts; a model waking from sleep reports ready again, and that
// is not a new load.
func (t *Tracker) ready(line string, now time.Time) *Outcome {
	port, ok := serverlog.LoadedPort(line)
	if !ok {
		port, ok = serverlog.ReadyPort(line)
	}
	if !ok {
		return nil
	}
	in := t.byPort[port]
	if in == nil || !in.readyAt.IsZero() {
		return nil
	}
	in.readyAt = now
	return &Outcome{
		Kind:        KindLoad,
		Time:        now,
		RouterName:  in.name,
		Port:        port,
		LoadSeconds: now.Sub(in.spawnedAt).Seconds(),
	}
}

// exited classifies the end of the instance named name.
//
// Exit status 1 is enough to see a crash. The router reports the status
// of a child that exited itself, and EXIT_FAILURE (1) for one killed by a
// signal (subprocess_join), so an abort after a CUDA error reads as 1.
// A stop that timed out and was killed reads as 1 as well, which is why
// the stop request, not the status, decides that case.
func (t *Tracker) exited(name string, status int, now time.Time) Outcome {
	var in *instance
	for _, c := range t.byPort {
		if c.name == name && (in == nil || c.spawnedAt.After(in.spawnedAt)) {
			in = c
		}
	}
	stopping := t.stopping[name]
	delete(t.stopping, name)

	out := Outcome{Time: now, RouterName: name, ExitStatus: status}
	if in == nil {
		// The spawn came before this tracker started, so whether it had
		// loaded is unknown. An instance that outlived a toolchest
		// restart had almost certainly loaded.
		if stopping || status == 0 {
			out.Kind = KindUnload
		} else {
			out.Kind = KindCrash
		}
		return out
	}

	delete(t.byPort, in.port)
	out.Port = in.port
	if !in.readyAt.IsZero() {
		out.LoadedSeconds = now.Sub(in.readyAt).Seconds()
	}
	switch {
	case stopping || status == 0:
		out.Kind = KindUnload
	case in.readyAt.IsZero():
		out.Kind = KindLoadFailed
	default:
		out.Kind = KindCrash
	}
	if out.Kind != KindUnload {
		out.Tail = append([]string(nil), in.tail...)
		out.ErrorLines = serverlog.ErrorLines(in.tail)
	}
	return out
}

// routerEnded returns the router's outcome followed by an unload for each
// instance still open, and forgets them all: the instances ended with
// their parent, not on their own, and the next router reuses ports.
func (t *Tracker) routerEnded(router Outcome) []Outcome {
	out := []Outcome{router}
	for _, in := range t.byPort {
		u := Outcome{Kind: KindUnload, Time: router.Time, RouterName: in.name, Port: in.port}
		if !in.readyAt.IsZero() {
			u.LoadedSeconds = router.Time.Sub(in.readyAt).Seconds()
		}
		out = append(out, u)
	}
	// Map order is random; the port order keeps the result the same
	// from one run to the next.
	sort.Slice(out[1:], func(i, j int) bool { return out[1+i].Port < out[1+j].Port })
	t.byPort = map[string]*instance{}
	t.stopping = map[string]bool{}
	return out
}

// Open returns the instances that have loaded and not yet ended. The
// slice is new; the caller may keep it.
func (t *Tracker) Open() []OpenInstance {
	var out []OpenInstance
	for _, in := range t.byPort {
		if !in.readyAt.IsZero() {
			out = append(out, OpenInstance{RouterName: in.name, Port: in.port, ReadySince: in.readyAt})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Port < out[j].Port })
	return out
}
