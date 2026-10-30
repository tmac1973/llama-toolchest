package enginenotes

import (
	"sort"
	"time"
)

// Summary is what happened in one period, for the strip, the Review
// dialog and the JSON endpoint.
type Summary struct {
	// CountingSince is when the saved record begins; From and To are the
	// period, From being no earlier than CountingSince.
	CountingSince time.Time `json:"counting_since"`
	From          time.Time `json:"from"`
	To            time.Time `json:"to"`

	Totals Totals     `json:"totals"`
	Models []ModelRow `json:"models"`
	// Failures are the crashes and failed loads, newest first.
	Failures []Event `json:"failures"`
}

// Totals counts the period's events by kind.
type Totals struct {
	RouterStartsUser int `json:"router_starts_user"`
	RouterStartsJob  int `json:"router_starts_job"`
	RouterStops      int `json:"router_stops"`
	RouterExits      int `json:"router_exits"`
	Loads            int `json:"loads"`
	FailedLoads      int `json:"failed_loads"`
	Crashes          int `json:"crashes"`
	Unloads          int `json:"unloads"`
	// FailuresInJobs is how many of the crashes and failed loads happened
	// while a job held the router.
	FailuresInJobs int `json:"failures_in_jobs"`
}

// ModelRow is one model's figures for the period.
type ModelRow struct {
	ModelID     string `json:"model_id"`
	Name        string `json:"name"`
	Loads       int    `json:"loads"`
	FailedLoads int    `json:"failed_loads"`
	Crashes     int    `json:"crashes"`
	Unloads     int    `json:"unloads"`
	// LoadedSeconds is time loaded: loads that ended in the period count
	// whole, and a model loaded now counts from when it loaded or from
	// the start of the period, whichever is later.
	LoadedSeconds  float64 `json:"loaded_seconds"`
	AvgLoadSeconds float64 `json:"avg_load_seconds"`
	// LastUsed is the later of the latest load and the start of the
	// latest hour with a request. Zero when neither is in the period.
	LastUsed       time.Time `json:"last_used,omitzero"`
	Requests       int       `json:"requests"`
	FailedRequests int       `json:"failed_requests"`
	PromptTokens   int       `json:"prompt_tokens"`
	GenTokens      int       `json:"gen_tokens"`
}

// Resolver names a model from the router's name for it (which is also its
// registry id): its registry id and display name. A model that no longer
// exists may come back as the name it was given.
type Resolver func(routerName string) (id, name string)

// Summary reports the last window of time, or everything saved when
// window is 0. open is the instances loaded now, whose time so far
// counts too.
func (s *Store) Summary(window time.Duration, open []OpenInstance, resolve Resolver) Summary {
	now := s.now().UTC()

	s.mu.Lock()
	defer s.mu.Unlock()

	from := s.data.CountingSince
	if window > 0 && now.Add(-window).After(from) {
		from = now.Add(-window)
	}
	sum := Summary{CountingSince: s.data.CountingSince, From: from, To: now}

	rows := map[string]*ModelRow{}
	loadTotals := map[string]float64{}
	row := func(id, name string) *ModelRow {
		r := rows[id]
		if r == nil {
			r = &ModelRow{ModelID: id}
			rows[id] = r
		}
		if r.Name == "" && name != "" {
			r.Name = name
		}
		return r
	}

	for _, e := range s.data.Events {
		if e.Time.Before(from) || e.Time.After(now) {
			continue
		}
		t := &sum.Totals
		switch e.Kind {
		case KindRouterStart:
			if e.StartReason == "job" {
				t.RouterStartsJob++
			} else {
				t.RouterStartsUser++
			}
			continue
		case KindRouterStop:
			t.RouterStops++
			continue
		case KindRouterExit:
			t.RouterExits++
			continue
		}

		id := e.ModelID
		if id == "" {
			id = e.RouterName
		}
		r := row(id, e.ModelName)
		switch e.Kind {
		case KindLoad:
			t.Loads++
			r.Loads++
			loadTotals[id] += e.LoadSeconds
			if e.Time.After(r.LastUsed) {
				r.LastUsed = e.Time
			}
		case KindLoadFailed:
			t.FailedLoads++
			r.FailedLoads++
		case KindCrash:
			t.Crashes++
			r.Crashes++
		case KindUnload:
			t.Unloads++
			r.Unloads++
		}
		r.LoadedSeconds += e.LoadedSeconds
		if e.IsFailure() {
			if e.JobID != "" {
				t.FailuresInJobs++
			}
			sum.Failures = append(sum.Failures, e)
		}
	}

	for _, in := range open {
		id, name := in.RouterName, ""
		if resolve != nil {
			id, name = resolve(in.RouterName)
		}
		since := in.ReadySince
		if since.Before(from) {
			since = from
		}
		if now.After(since) {
			row(id, name).LoadedSeconds += now.Sub(since).Seconds()
		}
	}

	for id, byHour := range s.data.Usage {
		for key, u := range byHour {
			// An hour counts when it starts in the period; usage is not
			// split within an hour.
			hour, err := time.Parse(hourKey, key)
			if err != nil || hour.Before(from) || hour.After(now) {
				continue
			}
			r := row(id, "")
			r.Requests += u.Requests
			r.FailedRequests += u.Failed
			r.PromptTokens += u.PromptTokens
			r.GenTokens += u.GenTokens
			if hour.After(r.LastUsed) {
				r.LastUsed = hour
			}
		}
	}

	for id, r := range rows {
		if r.Loads > 0 {
			r.AvgLoadSeconds = loadTotals[id] / float64(r.Loads)
		}
		if r.Name == "" {
			r.Name = id
			if resolve != nil {
				if _, name := resolve(id); name != "" {
					r.Name = name
				}
			}
		}
		sum.Models = append(sum.Models, *r)
	}
	sort.Slice(sum.Models, func(i, j int) bool { return sum.Models[i].Name < sum.Models[j].Name })
	sort.SliceStable(sum.Failures, func(i, j int) bool { return sum.Failures[i].Time.After(sum.Failures[j].Time) })
	return sum
}
