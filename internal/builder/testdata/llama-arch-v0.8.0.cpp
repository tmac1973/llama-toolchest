// Trimmed from llama.cpp v0.8.0 src/llama-arch.cpp: the LLM_ARCH_NAMES table,
// the tensor-split function, and the function after it (a similar switch
// that must not be read as part of it).

static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_CLIP,             "clip"             }, // dummy, only used by llama-quantize
    { LLM_ARCH_LLAMA,            "llama"            },
    { LLM_ARCH_LLAMA4,           "llama4"           },
    { LLM_ARCH_DECI,             "deci"             },
    { LLM_ARCH_FALCON,           "falcon"           },
    { LLM_ARCH_GROK,             "grok"             },
    { LLM_ARCH_GPT2,             "gpt2"             },
    { LLM_ARCH_GPTJ,             "gptj"             },
    { LLM_ARCH_GPTNEOX,          "gptneox"          },
    { LLM_ARCH_MPT,              "mpt"              },
    { LLM_ARCH_BAICHUAN,         "baichuan"         },
    { LLM_ARCH_STARCODER,        "starcoder"        },
    { LLM_ARCH_REFACT,           "refact"           },
    { LLM_ARCH_BERT,             "bert"             },
    { LLM_ARCH_MODERN_BERT,      "modern-bert"      },
    { LLM_ARCH_NOMIC_BERT,       "nomic-bert"       },
    { LLM_ARCH_NOMIC_BERT_MOE,   "nomic-bert-moe"   },
    { LLM_ARCH_NEO_BERT,         "neo-bert"         },
    { LLM_ARCH_JINA_BERT_V2,     "jina-bert-v2"     },
    { LLM_ARCH_JINA_BERT_V3,     "jina-bert-v3"     },
    { LLM_ARCH_EUROBERT,         "eurobert"         },
    { LLM_ARCH_BLOOM,            "bloom"            },
    { LLM_ARCH_STABLELM,         "stablelm"         },
    { LLM_ARCH_QWEN,             "qwen"             },
    { LLM_ARCH_QWEN2,            "qwen2"            },
    { LLM_ARCH_QWEN2MOE,         "qwen2moe"         },
    { LLM_ARCH_QWEN2VL,          "qwen2vl"          },
    { LLM_ARCH_SENSENOVA_U1,     "sensenova_u1"     },
    { LLM_ARCH_QWEN3,            "qwen3"            },
    { LLM_ARCH_FLINT,            "flint"            },
    { LLM_ARCH_QWEN3MOE,         "qwen3moe"         },
    { LLM_ARCH_QWEN3NEXT,        "qwen3next"        },
    { LLM_ARCH_QWEN3VL,          "qwen3vl"          },
    { LLM_ARCH_QWEN3VLMOE,       "qwen3vlmoe"       },
    { LLM_ARCH_QWEN35,           "qwen35"           },
    { LLM_ARCH_QWEN35MOE,        "qwen35moe"        },
    { LLM_ARCH_CLEF,             "clef"             },
    { LLM_ARCH_VLLM_DECISION2,   "vllm-decision2"   },
    { LLM_ARCH_QWEN4EXP,         "qwen4exp"         },
    { LLM_ARCH_PHI2,             "phi2"             },
    { LLM_ARCH_PHI3,             "phi3"             },
    { LLM_ARCH_PHIMOE,           "phimoe"           },
    { LLM_ARCH_PLAMO,            "plamo"            },
    { LLM_ARCH_PLAMO2,           "plamo2"           },
    { LLM_ARCH_PLAMO3,           "plamo3"           },
    { LLM_ARCH_CODESHELL,        "codeshell"        },
    { LLM_ARCH_ORION,            "orion"            },
    { LLM_ARCH_INTERNLM2,        "internlm2"        },
    { LLM_ARCH_MINICPM,          "minicpm"          },
    { LLM_ARCH_MINICPM3,         "minicpm3"         },
    { LLM_ARCH_GEMMA,            "gemma"            },
    { LLM_ARCH_GEMMA2,           "gemma2"           },
    { LLM_ARCH_GEMMA3,           "gemma3"           },
    { LLM_ARCH_GEMMA3N,          "gemma3n"          },
    { LLM_ARCH_GEMMA4,           "gemma4"           },
    { LLM_ARCH_GEMMA4_ASSISTANT, "gemma4-assistant" },
    { LLM_ARCH_GEMMA_EMBEDDING,  "gemma-embedding"  },
    { LLM_ARCH_GEMMA_EMBEDDING2, "gemma-embedding2" },
    { LLM_ARCH_STARCODER2,       "starcoder2"       },
    { LLM_ARCH_MAMBA,            "mamba"            },
    { LLM_ARCH_MAMBA2,           "mamba2"           },
    { LLM_ARCH_MAPLE,            "maple"            },
    { LLM_ARCH_JAMBA,            "jamba"            },
    { LLM_ARCH_FALCON_H1,        "falcon-h1"        },
    { LLM_ARCH_XVERSE,           "xverse"           },
    { LLM_ARCH_COMMAND_R,        "command-r"        },
    { LLM_ARCH_COHERE2,          "cohere2"          },
    { LLM_ARCH_COHERE2MOE,       "cohere2moe"       },
    { LLM_ARCH_DBRX,             "dbrx"             },
    { LLM_ARCH_OLMO,             "olmo"             },
    { LLM_ARCH_OLMO2,            "olmo2"            },
    { LLM_ARCH_OLMOE,            "olmoe"            },
    { LLM_ARCH_MUSE_GLIMMER,     "muse-glimmer"     },
    { LLM_ARCH_MUSE_SPARK,       "muse-spark"       },
    { LLM_ARCH_OPENELM,          "openelm"          },
    { LLM_ARCH_ARCTIC,           "arctic"           },
    { LLM_ARCH_DEEPSEEK,         "deepseek"         },
    { LLM_ARCH_DEEPSEEK2,        "deepseek2"        },
    { LLM_ARCH_DEEPSEEK2OCR,     "deepseek2-ocr"    },
    { LLM_ARCH_DEEPSEEK32,       "deepseek32"       },
    { LLM_ARCH_DEEPSEEK4,        "deepseek4"        },
    { LLM_ARCH_DEEPSEEK41,       "deepseek41"       },
    { LLM_ARCH_CHATGLM,          "chatglm"          },
    { LLM_ARCH_GLM4,             "glm4"             },
    { LLM_ARCH_GLM4_MOE,         "glm4moe"          },
    { LLM_ARCH_GLM_DSA,          "glm-dsa"          },
    { LLM_ARCH_BITNET,           "bitnet"           },
    { LLM_ARCH_T5,               "t5"               },
    { LLM_ARCH_T5ENCODER,        "t5encoder"        },
    { LLM_ARCH_JAIS,             "jais"             },
    { LLM_ARCH_JAIS2,            "jais2"            },
    { LLM_ARCH_NEMOTRON,         "nemotron"         },
    { LLM_ARCH_NEMOTRON_H,       "nemotron_h"       },
    { LLM_ARCH_NEMOTRON_H_MOE,   "nemotron_h_moe"   },
    { LLM_ARCH_EXAONE,           "exaone"           },
    { LLM_ARCH_EXAONE4,          "exaone4"          },
    { LLM_ARCH_EXAONE_MOE,       "exaone-moe"       },
    { LLM_ARCH_RWKV6,            "rwkv6"            },
    { LLM_ARCH_RWKV6QWEN2,       "rwkv6qwen2"       },
    { LLM_ARCH_RWKV7,            "rwkv7"            },
    { LLM_ARCH_ARWKV7,           "arwkv7"           },
    { LLM_ARCH_GRANITE,          "granite"          },
    { LLM_ARCH_GRANITE_MOE,      "granitemoe"       },
    { LLM_ARCH_GRANITE_HYBRID,   "granitehybrid"    },
    { LLM_ARCH_GRANITE_SWITCH,   "graniteswitch"    },
    { LLM_ARCH_GRANITE_SWA,      "granite_swa"      },
    { LLM_ARCH_CHAMELEON,        "chameleon"        },
    { LLM_ARCH_WAVTOKENIZER_DEC, "wavtokenizer-dec" },
    { LLM_ARCH_PLM,              "plm"              },
    { LLM_ARCH_BAILINGMOE,       "bailingmoe"       },
    { LLM_ARCH_BAILINGMOE2,      "bailingmoe2"      },
    { LLM_ARCH_BAILINGMOE3,      "bailingmoe3"      },
    { LLM_ARCH_DOTS1,            "dots1"            },
    { LLM_ARCH_DOTS3NOTE,        "dots3note"        },
    { LLM_ARCH_ARCEE,            "arcee"            },
    { LLM_ARCH_AFMOE,            "afmoe"            },
    { LLM_ARCH_LAGUNA,           "laguna"           },
    { LLM_ARCH_ERNIE4_5,         "ernie4_5"         },
    { LLM_ARCH_ERNIE4_5_MOE,     "ernie4_5-moe"     },
    { LLM_ARCH_HUNYUAN_MOE,      "hunyuan-moe"      },
    { LLM_ARCH_HUNYUAN_DENSE,    "hunyuan-dense"    },
    { LLM_ARCH_HUNYUAN_VL,       "hunyuan_vl"       },
    { LLM_ARCH_HY_V3,            "hy_v3"            },
    { LLM_ARCH_HY_V4,            "hy_v4"            },
    { LLM_ARCH_SMOLLM3,          "smollm3"          },
    { LLM_ARCH_OPENAI_MOE,       "gpt-oss"          },
    { LLM_ARCH_LFM2,             "lfm2"             },
    { LLM_ARCH_LFM2MOE,          "lfm2moe"          },
    { LLM_ARCH_LFM3,             "lfm3"             },
    { LLM_ARCH_DREAM,            "dream"            },
    { LLM_ARCH_SMALLTHINKER,     "smallthinker"     },
    { LLM_ARCH_LLADA,            "llada"            },
    { LLM_ARCH_LLADA_MOE,        "llada-moe"        },
    { LLM_ARCH_SEED_OSS,         "seed_oss"         },
    { LLM_ARCH_GROVEMOE,         "grovemoe"         },
    { LLM_ARCH_APERTUS,          "apertus"          },
    { LLM_ARCH_MINIMAX_01,       "minimax-01"       },
    { LLM_ARCH_HRM_TEXT,         "hrm_text"         },
    { LLM_ARCH_MINIMAX_M2,       "minimax-m2"       },
    { LLM_ARCH_MINIMAX_M3,       "minimax-m3"       },
    { LLM_ARCH_COGVLM,           "cogvlm"           },
    { LLM_ARCH_RND1,             "rnd1"             },
    { LLM_ARCH_PANGU_EMBED,      "pangu-embedded"   },
    { LLM_ARCH_MISTRAL3,         "mistral3"         },
    { LLM_ARCH_EAGLE3,           "eagle3"           },
    { LLM_ARCH_DFLASH,           "dflash"           },
    { LLM_ARCH_MISTRAL4,         "mistral4"         },
    { LLM_ARCH_PADDLEOCR,        "paddleocr"        },
    { LLM_ARCH_MIMO2,            "mimo2"            },
    { LLM_ARCH_STEP35,           "step35"           },
    { LLM_ARCH_SPARK2_5,         "spark2_5"         },
    { LLM_ARCH_LLAMA_EMBED,      "llama-embed"      },
    { LLM_ARCH_MAINCODER,        "maincoder"        },
    { LLM_ARCH_MAION_CODER,      "maion-coder"      },
    { LLM_ARCH_KIMI_LINEAR,      "kimi-linear"      },
    { LLM_ARCH_KIMI_K3,          "kimi-k3"          },
    { LLM_ARCH_GLM5_NEXT,        "glm5-next"        },
    { LLM_ARCH_TALKIE,           "talkie"           },
    { LLM_ARCH_MELLUM,           "mellum"           },
    { LLM_ARCH_NANBEIGE,         "nanbeige"         },
    { LLM_ARCH_QWEN3TTS,         "qwen3tts"         },
    { LLM_ARCH_POCKETTTS,        "pockettts"        },
    { LLM_ARCH_K2_HORIZON,       "k2-horizon"       },
    { LLM_ARCH_BEAM,             "beam"             },
    { LLM_ARCH_KOLIBRI1,         "kolibri1"         },
    { LLM_ARCH_UNKNOWN,          "(unknown)"        },
};

bool llm_arch_supports_sm_tensor(const llm_arch & arch) {
    switch (arch) {
        case LLM_ARCH_GROK:
        case LLM_ARCH_MPT:
        case LLM_ARCH_PLAMO2:
        case LLM_ARCH_MINICPM3:
        case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
        case LLM_ARCH_JAMBA:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_T5:
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
        case LLM_ARCH_GRANITE_HYBRID:
        case LLM_ARCH_LFM3:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM: // [TAG_MOE_AMAX_SCALE]
        case LLM_ARCH_FLINT:  // [TAG_MOE_AMAX_SCALE]
        case LLM_ARCH_LAGUNA: // [TAG_MOE_AMAX_SCALE]
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_QWEN3TTS:
            return false;
        default:
            return true;
    }
}

bool llama_arch_supports_save(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_SENSENOVA_U1:
        case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_T5:
        case LLM_ARCH_APERTUS:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_CLEF: // the head tensors are not saved
        case LLM_ARCH_VLLM_DECISION2:
            return false;
        default:
            return true;
    }
}

