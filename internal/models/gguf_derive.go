package models

// unknownExpertShare is the part of an MoE file assumed to be expert
// weights when the parameter count is not known. Expert tensors are the
// great majority of every MoE model checked (91% of gpt-oss-20b's
// parameters); the figure only sets how much system memory an expert
// offload is reported to need.
const unknownExpertShare = 0.9

// DerivedFor returns a copy of a metadata-only description (see
// ParseGGUFMetaOnly) with the tensor sizes the VRAM estimate needs worked
// out for one file of fileBytes, instead of measured from its tensor table.
//
// The quants in a repository all describe the same model, so one header
// read serves every file; only the byte sizes differ. Each size is the
// tensor's share of the model's parameters, applied to the file size.
// paramCount is the whole model's parameter count (HuggingFace's
// gguf.total), or 0 when it is not known.
//
// The share is of parameters, not bytes. Quantizers often keep the
// embedding and attention tensors at more bits than the experts, so a
// byte share differs from the parameter share by a few percent either
// way; the test against full parses of real files bounds the effect.
//
// A description that was read in full is returned unchanged: its sizes
// are measured.
func (meta *GGUFMeta) DerivedFor(fileBytes, paramCount int64) *GGUFMeta {
	d := *meta
	if !meta.MetaOnly || fileBytes <= 0 {
		return &d
	}
	share := func(params int64) int64 {
		if paramCount <= 0 || params <= 0 {
			return 0
		}
		return int64(float64(fileBytes) * min(1, float64(params)/float64(paramCount)))
	}

	if d.ExpertCount > 0 && d.NLayers > 0 {
		// --n-cpu-moe counts layers from 0, so the dense layers some MoE
		// models begin with come first. A trailing NextN (MTP) layer
		// carries experts too; llama.cpp counts it among block_count.
		d.ExpertLayerFirst = min(d.LeadingDenseBlocks, d.NLayers)
		d.ExpertLayers = d.NLayers - d.ExpertLayerFirst
		if d.ExpertFFLength > 0 && d.NEmbd > 0 && paramCount > 0 {
			// gate, up and down: three matrices of n_embd × ff per expert.
			params := int64(d.ExpertLayers) * int64(d.ExpertCount) * 3 * int64(d.NEmbd) * int64(d.ExpertFFLength)
			d.ExpertBytes = share(params)
		} else {
			d.ExpertBytes = int64(float64(fileBytes) * unknownExpertShare)
		}
	}
	if d.VocabSize > 0 && d.NEmbd > 0 {
		d.TokenEmbdBytes = share(int64(d.VocabSize) * int64(d.NEmbd))
	}
	if d.PLEInputDim > 0 && d.VocabSize > 0 {
		d.PLEBytes = share(int64(d.VocabSize) * int64(d.NLayers) * int64(d.PLEInputDim))
	}
	return &d
}
