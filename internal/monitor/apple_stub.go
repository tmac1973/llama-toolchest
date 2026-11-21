//go:build !darwin

package monitor

func newApple() GPUBackend { return nil }
