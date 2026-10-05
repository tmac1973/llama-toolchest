//go:build linux

package monitor

import (
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

// fakeNVIDIASMI puts an nvidia-smi script first on PATH that prints out
// and exits with code, and records the arguments it was called with in
// the returned file. Collect has no other seam: it runs nvidia-smi by name.
func fakeNVIDIASMI(t *testing.T, out string, code int) (argsFile string) {
	t.Helper()
	dir := t.TempDir()
	outFile := filepath.Join(dir, "out.csv")
	argsFile = filepath.Join(dir, "args.txt")
	if err := os.WriteFile(outFile, []byte(out), 0o644); err != nil {
		t.Fatal(err)
	}
	script := "#!/bin/sh\n" +
		"printf '%s\\n' \"$@\" > '" + argsFile + "'\n" +
		"cat '" + outFile + "'\n" +
		"exit " + strconv.Itoa(code) + "\n"
	if err := os.WriteFile(filepath.Join(dir, "nvidia-smi"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	return argsFile
}

// The parser reads fields by position, so it only works if the query asks
// for them in that order and without units ("16376", not "16376 MiB").
// This pins the two together.
func TestNVIDIACollectQueriesFieldsInParseOrder(t *testing.T) {
	argsFile := fakeNVIDIASMI(t, "", 0)
	if _, err := (&nvidiaBacked{}).Collect(); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Fields(string(raw))
	want := []string{
		"--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed",
		"--format=csv,noheader,nounits",
	}
	if !reflect.DeepEqual(args, want) {
		t.Errorf("nvidia-smi called with %q, want %q", args, want)
	}
}

// What nvidia-smi prints for the query, and what Collect must make of it.
// The first case is real output from a host with three RTX A4000 cards.
// A passively cooled card reports its fan as "[N/A]", which must read as
// "no fan" rather than a stopped fan; other "[N/A]" fields read as 0.
// Rows too short to hold the required fields, and blank lines, are
// skipped rather than turned into a zero GPU.
func TestNVIDIACollectParsesCSV(t *testing.T) {
	a4000 := func(idx int, temp int, power float64) GPUInfo {
		return GPUInfo{Index: idx, Name: "NVIDIA RTX A4000", VRAMUsedMB: 4, VRAMTotalMB: 16376,
			TempC: temp, PowerW: power, HasFan: true, FanPercent: 41}
	}
	cases := []struct {
		name string
		out  string
		want []GPUInfo
	}{
		{
			name: "three GPUs, real output",
			out: "0, NVIDIA RTX A4000, 0, 4, 16376, 37, 14.09, 41\n" +
				"1, NVIDIA RTX A4000, 0, 4, 16376, 37, 15.14, 41\n" +
				"2, NVIDIA RTX A4000, 0, 4, 16376, 39, 11.83, 41\n",
			want: []GPUInfo{a4000(0, 37, 14.09), a4000(1, 37, 15.14), a4000(2, 39, 11.83)},
		},
		{
			name: "passive card reports no fan",
			out:  "0, NVIDIA A100-PCIE-40GB, 87, 30001, 40960, 52, 211.40, [N/A]\n",
			want: []GPUInfo{{Index: 0, Name: "NVIDIA A100-PCIE-40GB", UtilPercent: 87, VRAMUsedMB: 30001,
				VRAMTotalMB: 40960, TempC: 52, PowerW: 211.4}},
		},
		{
			name: "fan stopped at 0 percent is still a fan",
			out:  "0, NVIDIA GeForce RTX 4090, 0, 300, 24564, 34, 21.00, 0\n",
			want: []GPUInfo{{Index: 0, Name: "NVIDIA GeForce RTX 4090", VRAMUsedMB: 300, VRAMTotalMB: 24564,
				TempC: 34, PowerW: 21, HasFan: true}},
		},
		{
			name: "N/A in other fields reads as 0",
			out:  "0, NVIDIA GH200 480GB, [N/A], [N/A], 97871, 40, N/A, [N/A]\n",
			want: []GPUInfo{{Index: 0, Name: "NVIDIA GH200 480GB", VRAMTotalMB: 97871, TempC: 40}},
		},
		{
			name: "row without the fan field",
			out:  "0, Tesla T4, 5, 100, 15360, 45, 27.5\n",
			want: []GPUInfo{{Index: 0, Name: "Tesla T4", UtilPercent: 5, VRAMUsedMB: 100, VRAMTotalMB: 15360,
				TempC: 45, PowerW: 27.5}},
		},
		{
			// nvidia-smi prints some warnings, such as a corrupted infoROM,
			// on standard output among the rows.
			name: "short rows, warnings and blank lines are skipped",
			out: "\n0, NVIDIA RTX A4000, 0, 4, 16376, 37, 14.09, 41\n" +
				"WARNING: infoROM is corrupted at gpu 0000:01:00.0\n" +
				"\n" +
				"1, NVIDIA RTX A4000, 0, 4\n" +
				"   \n" +
				"2, NVIDIA RTX A4000, 0, 4, 16376, 39, 11.83, 41\n\n",
			want: []GPUInfo{a4000(0, 37, 14.09), a4000(2, 39, 11.83)},
		},
		{
			name: "extra spaces and CRLF line ends are trimmed",
			out:  "  0 ,  NVIDIA RTX A4000 ,  0 ,  4 ,  16376 ,  37 ,  14.09 ,  41  \r\n1, NVIDIA RTX A4000, 0, 4, 16376, 37, 15.14, 41\r\n",
			want: []GPUInfo{a4000(0, 37, 14.09), a4000(1, 37, 15.14)},
		},
		{name: "no output", out: "", want: nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fakeNVIDIASMI(t, c.out, 0)
			got, err := (&nvidiaBacked{}).Collect()
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("Collect =\n %+v\nwant\n %+v", got, c.want)
			}
		})
	}
}

// A failing nvidia-smi (driver unloaded, GPU fell off the bus) must be an
// error, not an empty GPU list that would look like "no GPUs".
func TestNVIDIACollectReportsFailure(t *testing.T) {
	fakeNVIDIASMI(t, "Unable to determine the device handle for GPU0000:01:00.0: Unknown Error\n", 1)
	if got, err := (&nvidiaBacked{}).Collect(); err == nil {
		t.Errorf("Collect = %+v, nil; want an error", got)
	}
}
