//go:build !darwin

package monitor

// perfCores is 0 off macOS: the planner then counts logical cores.
func perfCores() int { return 0 }
