package modelsource

import (
	"sort"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

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
		base, total, isShard := models.SplitShardName(f.Filename)
		if !isShard {
			singles = append(singles, f)
			continue
		}
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
			VRAMEstGB:  models.EstimateVRAM(totalSize),
			Shards:     shardNames,
			ShardSizes: shardSizes,
			OID:        g.shards[0].OID,
		})
	}

	// Sort grouped entries by filename for stable ordering
	sort.Slice(result, func(i, j int) bool {
		return result[i].Filename < result[j].Filename
	})

	return append(result, singles...)
}
