package modelsource

import (
	"regexp"
	"sort"
	"strconv"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// EstimateVRAM returns estimated VRAM in GB.
// Uses file size * 1.1 as a rough estimate (overhead for KV cache and buffers).
func EstimateVRAM(sizeBytes int64) float64 {
	return float64(sizeBytes) * 1.1 / (1024 * 1024 * 1024)
}

// shardPattern matches split GGUF filenames like "model-00001-of-00005.gguf"
var shardPattern = regexp.MustCompile(`^(.+)-(\d{5})-of-(\d{5})\.gguf$`)

// GroupShards merges split GGUF shard files into single entries.
// e.g., 5 files "model-0000N-of-00005.gguf" become one entry with combined size.
func GroupShards(files []File) []File {
	type shardGroup struct {
		base   string
		total  int
		shards []File
	}
	groups := map[string]*shardGroup{}
	var singles []File

	for _, f := range files {
		m := shardPattern.FindStringSubmatch(f.Filename)
		if m == nil {
			singles = append(singles, f)
			continue
		}
		base := m[1]
		total, _ := strconv.Atoi(m[3])
		g, ok := groups[base]
		if !ok {
			g = &shardGroup{base: base, total: total}
			groups[base] = g
		}
		g.shards = append(g.shards, f)
	}

	var result []File
	for _, g := range groups {
		sort.Slice(g.shards, func(i, j int) bool {
			return g.shards[i].Filename < g.shards[j].Filename
		})
		var totalSize int64
		var shardNames []string
		var shardSizes []int64
		for _, s := range g.shards {
			totalSize += s.Size
			shardNames = append(shardNames, s.Filename)
			shardSizes = append(shardSizes, s.Size)
		}
		result = append(result, File{
			Filename:   g.shards[0].Filename,
			Size:       totalSize,
			Quant:      g.shards[0].Quant,
			VRAMEstGB:  EstimateVRAM(totalSize),
			Shards:     shardNames,
			ShardSizes: shardSizes,
		})
	}

	// Sort grouped entries by filename for stable ordering
	sort.Slice(result, func(i, j int) bool {
		return result[i].Filename < result[j].Filename
	})

	return append(result, singles...)
}

// ExpandShards returns all shard filenames for a split GGUF, or a
// single-element slice for non-split files. The naming rule lives in
// models, which needs it to find a split model's tensor table.
func ExpandShards(filename string) []string { return models.ExpandShards(filename) }
