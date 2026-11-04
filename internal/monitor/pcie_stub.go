//go:build !linux

package monitor

func readPCIeLink(string) *PCIeLink { return nil }
