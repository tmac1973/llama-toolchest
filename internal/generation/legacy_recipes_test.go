package generation

// The recipe tables toolchest kept by hand before it read upstream's
// recipe files (copied from stable-diffusion.cpp at master-1000-95ff762,
// size multiples at master-1015-e41fbcf). TestSnapshotMatchesLegacyTables
// checks that the snapshot gives the same slots, size multiples and
// capabilities for these recipes.

type legacyVideoRecipe struct {
	Slots        RecipeSlotSet
	Caps         []string
	SizeMultiple int
}

var legacyImageRecipes = map[string]RecipeSlotSet{
	"ming-image":           {Required: []string{"diffusion_model", "llm", "tokenizer"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}},
	"qwen-image-21-turbo":  {Required: []string{"diffusion_model", "llm"}, Optional: []string{"llm_vision", "vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}},
	"qwen-image-21":        {Required: []string{"diffusion_model", "llm"}, Optional: []string{"llm_vision", "vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}},
	"flux2-klein-4b":       {Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}},
	"z-image-turbo":        {Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}},
	"flux3-klein":          {Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}},
	"qwen-image-edit-2511": {Required: []string{"diffusion_model", "llm"}, Optional: []string{"llm_vision", "vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}},
	"z-image":              {Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}},
	"chroma":               {Required: []string{"diffusion_model", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}},
	"flux1-dev":            {Required: []string{"diffusion_model", "clip_l", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}},
	"flux1-schnell":        {Required: []string{"diffusion_model", "clip_l", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}},
	"sdxl-checkpoint":      {Required: []string{"model"}, Optional: []string{"vae", "tae"}},
}

var legacyVideoRecipes = map[string]legacyVideoRecipe{
	"wan22-ti2v-5b":        {legacyWanSlots(), []string{CapText, CapStartImage}, 32},
	"wan22-i2v":            {legacyWanMoESlots(), []string{CapStartImage, CapEndImage}, 16},
	"wan22-t2v":            {legacyWanMoESlots(), []string{CapText}, 16},
	"wan21-t2v":            {legacyWanSlots(), []string{CapText}, 16},
	"wan21-i2v":            {RecipeSlotSet{Required: []string{"diffusion_model", "t5xxl"}, Optional: []string{"vae", "tae", "clip_vision"}, OneOf: [][]string{{"vae", "tae"}}}, []string{CapStartImage}, 16},
	"wan21-flf2v":          {RecipeSlotSet{Required: []string{"diffusion_model", "t5xxl", "clip_vision"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}}, []string{CapStartImage, CapEndImage}, 16},
	"ltx23-distilled":      {legacyLtxSlots(), []string{CapText, CapStartImage, CapEndImage}, 32},
	"ltx23-dev":            {legacyLtxSlots(), []string{CapText, CapStartImage, CapEndImage}, 32},
	"ltx25-dev":            {RecipeSlotSet{Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae", "audio_vae"}, OneOf: [][]string{{"vae", "tae"}}}, []string{CapText, CapStartImage, CapEndImage}, 32},
	"hunyuan-video-15-t2v": {legacyHunyuanSlots(), []string{CapText}, 16},
	"hunyuan-video-15-i2v": {legacyHunyuanSlots(), []string{CapStartImage}, 16},
	"lingbot-video-dense":  {legacyLlmVideoSlots(), []string{CapText, CapStartImage}, 16},
	"lingbot-video-moe":    {legacyLlmVideoSlots(), []string{CapText, CapStartImage}, 16},
}

func legacyWanSlots() RecipeSlotSet {
	return RecipeSlotSet{Required: []string{"diffusion_model", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}}
}

func legacyWanMoESlots() RecipeSlotSet {
	return RecipeSlotSet{Required: []string{"diffusion_model", "high_noise_diffusion_model", "t5xxl"},
		Optional: []string{"vae", "tae", "lora", "high_noise_lora"}, OneOf: [][]string{{"vae", "tae"}}}
}

func legacyLtxSlots() RecipeSlotSet {
	return RecipeSlotSet{Required: []string{"diffusion_model", "llm", "embeddings_connectors"}, Optional: []string{"vae", "tae", "audio_vae"}, OneOf: [][]string{{"vae", "tae"}}}
}

func legacyHunyuanSlots() RecipeSlotSet {
	return RecipeSlotSet{Required: []string{"diffusion_model", "llm", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}}
}

func legacyLlmVideoSlots() RecipeSlotSet {
	return RecipeSlotSet{Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: [][]string{{"vae", "tae"}}}
}
