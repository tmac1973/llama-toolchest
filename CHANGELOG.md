# Changelog

## [2.16.3](https://github.com/tmac1973/llama-toolchest/compare/v2.16.2...v2.16.3) (2026-07-24)


### Bug Fixes

* **obs:** show a single averaged row per model in live performance ([#95](https://github.com/tmac1973/llama-toolchest/issues/95)) ([e333e10](https://github.com/tmac1973/llama-toolchest/commit/e333e10698de2d3a269d7e5fe01b01e60272a08c))

## [2.16.2](https://github.com/tmac1973/llama-toolchest/compare/v2.16.1...v2.16.2) (2026-07-19)


### Bug Fixes

* **obs:** report benchmark and live throughput per prompt size, matching llama-bench ([#93](https://github.com/tmac1973/llama-toolchest/issues/93)) ([feadda3](https://github.com/tmac1973/llama-toolchest/commit/feadda317fe353a71b6930276ba459d3ad84abcd))

## [2.16.1](https://github.com/tmac1973/llama-toolchest/compare/v2.16.0...v2.16.1) (2026-07-19)


### Bug Fixes

* **bench:** prompt-cache contamination across prompt sizes, and batch/ubatch in results UI ([#91](https://github.com/tmac1973/llama-toolchest/issues/91)) ([9ae7371](https://github.com/tmac1973/llama-toolchest/commit/9ae7371bdce6e0e14d677af6988bbe9367ba24b5))

## [2.16.0](https://github.com/tmac1973/llama-toolchest/compare/v2.15.1...v2.16.0) (2026-07-18)


### Features

* **bench:** parameter sweeps, working config overrides, and ROCm flag corrections ([#90](https://github.com/tmac1973/llama-toolchest/issues/90)) ([359bf4b](https://github.com/tmac1973/llama-toolchest/commit/359bf4b4d790be00cc2c11b003bb351ce1a405cb))


### Bug Fixes

* **build:** correct Go module path to github.com/tmac1973/llama-toolchest ([#88](https://github.com/tmac1973/llama-toolchest/issues/88)) ([f7871d7](https://github.com/tmac1973/llama-toolchest/commit/f7871d7a0b7b873977dadbf5ebf1362e66e4e0e2))

## [2.15.1](https://github.com/tmac1973/llama-toolchest/compare/v2.15.0...v2.15.1) (2026-07-15)


### Bug Fixes

* **ui:** drop sidebar brand badge, enlarge sidebar text ([#86](https://github.com/tmac1973/llama-toolchest/issues/86)) ([74c2949](https://github.com/tmac1973/llama-toolchest/commit/74c29493a7344fb60c29bd0081dbbb1886640916))

## [2.15.0](https://github.com/tmac1973/llama-toolchest/compare/v2.14.0...v2.15.0) (2026-07-15)


### Features

* **models:** recognize FP4 quants (ROCmFP4, NVFP4, bare FP4) ([#83](https://github.com/tmac1973/llama-toolchest/issues/83)) ([53d2d8a](https://github.com/tmac1973/llama-toolchest/commit/53d2d8a30168f33f8f7865ece80056e46b128580))
* **ui:** shared UI refresh + optional Graphite theme ([#85](https://github.com/tmac1973/llama-toolchest/issues/85)) ([bc64754](https://github.com/tmac1973/llama-toolchest/commit/bc64754d7ddf269ebadc3ddf81dfd3ecbf73d94a))


### Bug Fixes

* **dashboard:** keep toggled-off models in the Available list until restart ([#82](https://github.com/tmac1973/llama-toolchest/issues/82)) ([220315d](https://github.com/tmac1973/llama-toolchest/commit/220315d19fd9cf632f465ce5920443565f0e9ca0))

## [2.14.0](https://github.com/tmac1973/llama-toolchest/compare/v2.13.1...v2.14.0) (2026-07-10)


### Features

* **builder:** add HIP Fast Math toggle to ROCm build options ([#80](https://github.com/tmac1973/llama-toolchest/issues/80)) ([2decf1c](https://github.com/tmac1973/llama-toolchest/commit/2decf1c07f3976ea83db46d21e11f9efdf8c5d40))


### Refactors

* eliminate code duplication found by the duplication audit ([#79](https://github.com/tmac1973/llama-toolchest/issues/79)) ([0a01e47](https://github.com/tmac1973/llama-toolchest/commit/0a01e47f520ad65a11ceb45ceed8f0e219c40da9))

## [2.13.1](https://github.com/tmac1973/llama-toolchest/compare/v2.13.0...v2.13.1) (2026-06-27)


### Bug Fixes

* **dashboard:** only list router-known models as available ([#77](https://github.com/tmac1973/llama-toolchest/issues/77)) ([5ad4c35](https://github.com/tmac1973/llama-toolchest/commit/5ad4c3503f7d3db250405e33c9f69a89ee5d8585))

## [2.13.0](https://github.com/tmac1973/llama-toolchest/compare/v2.12.0...v2.13.0) (2026-06-22)


### Features

* optional secure deploy (Caddy reverse proxy) + scriptable installer ([622ec3d](https://github.com/tmac1973/llama-toolchest/commit/622ec3d785b6ebb6823afdc2d5b0c0fc93c65f15))
* **setup:** optional secure deploy (Caddy reverse proxy) + scriptable installer ([1bbcfa5](https://github.com/tmac1973/llama-toolchest/commit/1bbcfa5409e53d7fe40636b65c71a38cb134a9fb))

## [2.12.0](https://github.com/tmac1973/llama-toolchest/compare/v2.11.0...v2.12.0) (2026-06-20)


### Features

* **api:** expose model capabilities for client auto-discovery ([241a9f4](https://github.com/tmac1973/llama-toolchest/commit/241a9f4465e65a6b68eaad5d2a3c0288bc2e373c))
* expose model capabilities for client auto-discovery ([107ea5a](https://github.com/tmac1973/llama-toolchest/commit/107ea5ab46fd32744aa16843e4ef8d41a71df8d1))

## [2.11.0](https://github.com/tmac1973/llama-toolchest/compare/v2.10.2...v2.11.0) (2026-06-19)


### Features

* **models:** surface incomplete downloads with in-place resume ([701221c](https://github.com/tmac1973/llama-toolchest/commit/701221c87f946aa78490807ca6b5792bac623314))


### Bug Fixes

* **config:** discover system config at /etc/llama-toolchest ([d332764](https://github.com/tmac1973/llama-toolchest/commit/d3327640252dd9250f36d8fa921e3d5755a23c51))
* correct GPU detection — name by Device Type + pin CUDA order ([#68](https://github.com/tmac1973/llama-toolchest/issues/68)) ([7c6081e](https://github.com/tmac1973/llama-toolchest/commit/7c6081ee9c345908db358eb909c2b302903baad8))
* discover system config at /etc/llama-toolchest ([#61](https://github.com/tmac1973/llama-toolchest/issues/61)) ([62e4868](https://github.com/tmac1973/llama-toolchest/commit/62e48688c190d50af51e825ad9bdb6435ed4a144))
* **monitor:** label ROCm GPUs by Device Type, not CPU-name blacklist ([aeaf79c](https://github.com/tmac1973/llama-toolchest/commit/aeaf79c331fadd113909771e2968895fc1be19cd))
* parse extra flags shell-style to preserve spaces ([#67](https://github.com/tmac1973/llama-toolchest/issues/67)) ([529a379](https://github.com/tmac1973/llama-toolchest/commit/529a3791928534cb22fb3c4d487c84fd8979253d))
* **preset:** parse extra flags shell-style to preserve spaces ([c4be663](https://github.com/tmac1973/llama-toolchest/commit/c4be6637f1c3b120361270d0c0a5a1d92e5977f0))
* **process:** pin CUDA_DEVICE_ORDER=PCI_BUS_ID for llama-server ([6d0fbd0](https://github.com/tmac1973/llama-toolchest/commit/6d0fbd0f9cc422998b1cab8e1b50bc997d9f7d15))

## [2.10.2](https://github.com/tmac1973/llama-toolchest/compare/v2.10.1...v2.10.2) (2026-06-09)


### Bug Fixes

* **quant:** re-derive persisted quant on registry load ([b7a2fcd](https://github.com/tmac1973/llama-toolchest/commit/b7a2fcd7b9fe24c46aa8dc84f824ca6ee1be07b6))
* **quant:** re-derive persisted quant on registry load ([d36d818](https://github.com/tmac1973/llama-toolchest/commit/d36d818e5e56311eff0e0c6469287b0a83b7aeb3))

## [2.10.1](https://github.com/tmac1973/llama-toolchest/compare/v2.10.0...v2.10.1) (2026-06-09)


### Bug Fixes

* **compose:** pin image name so Quadlet unit can find the built image ([cc4b05f](https://github.com/tmac1973/llama-toolchest/commit/cc4b05f67a3c1a86a81b270c70012ec189dd67d3))
* **compose:** pin image name so Quadlet unit can find the built image ([d1cf27d](https://github.com/tmac1973/llama-toolchest/commit/d1cf27d4430a3f7f91a245f1dc1ea02b8b5969b1))
* **quant:** parse UD prefix and suffixes consistently, recognize MXFP4 ([feca9fc](https://github.com/tmac1973/llama-toolchest/commit/feca9fca0b2a7c3bab22d604a456492c93000896))
* **quant:** parse UD prefix and suffixes consistently, recognize MXFP4 ([3df08ab](https://github.com/tmac1973/llama-toolchest/commit/3df08ab9c14e0a5838c4c820e979217df915a12d))

## [2.10.0](https://github.com/tmac1973/llama-toolchest/compare/v2.9.0...v2.10.0) (2026-06-08)


### Features

* **mtp:** support gemma-4 separate MTP drafter heads ([9b9194b](https://github.com/tmac1973/llama-toolchest/commit/9b9194b7a5f3d9d2ffbff94fb9de108756fceea0))
* **mtp:** support gemma-4 separate MTP drafter heads ([3821601](https://github.com/tmac1973/llama-toolchest/commit/382160104ffd1f1e513c832873190f2997611457))


### Bug Fixes

* **vram:** model GQA + sliding-window attention and auxiliary files ([c9849e7](https://github.com/tmac1973/llama-toolchest/commit/c9849e78b8fa5370a79cfdd249eda52d689ae980))
* **vram:** model GQA + sliding-window attention and auxiliary files ([b8720eb](https://github.com/tmac1973/llama-toolchest/commit/b8720ebea55b1c51a0825cdf8924280915cf7182))

## [2.9.0](https://github.com/tmac1973/llama-toolchest/compare/v2.8.0...v2.9.0) (2026-06-06)


### Features

* **benchmark:** add fp16 and MTP overrides to batch job options ([492827c](https://github.com/tmac1973/llama-toolchest/commit/492827ce0623a55325daeb1f8166d642ae1fd5de))
* **benchmark:** add fp16 and MTP overrides to batch job options ([fa5a3f6](https://github.com/tmac1973/llama-toolchest/commit/fa5a3f6053b0179c3667664618a7f883bb45d1fd))

## [2.8.0](https://github.com/tmac1973/llama-toolchest/compare/v2.7.0...v2.8.0) (2026-05-29)


### Features

* **mmproj:** add toggle to skip vision projector without clearing path ([3ae429f](https://github.com/tmac1973/llama-toolchest/commit/3ae429f5d587267e51e2829849e31c5258fb1755))
* **mmproj:** add toggle to skip vision projector without clearing path ([44cbde3](https://github.com/tmac1973/llama-toolchest/commit/44cbde3dd82b1a6849f955221f9d905ba83dde58))

## [2.7.0](https://github.com/tmac1973/llama-toolchest/compare/v2.6.1...v2.7.0) (2026-05-29)


### Features

* **rocm:** add FP16 cuBLAS compute toggle and bump ROCm to 7.2.4 ([1f66bcd](https://github.com/tmac1973/llama-toolchest/commit/1f66bcde13c0e2db45ad48d7c7dd94efef964263))
* **rocm:** add FP16 cuBLAS compute toggle and bump ROCm to 7.2.4 ([c638f22](https://github.com/tmac1973/llama-toolchest/commit/c638f2240c4be8cbcf54229ae83ec9f41cb4840e))

## [2.6.1](https://github.com/tmac1973/llama-toolchest/compare/v2.6.0...v2.6.1) (2026-05-25)


### Bug Fixes

* **cuda:** pin libnccl-dev to installed libnccl2 version ([a12273f](https://github.com/tmac1973/llama-toolchest/commit/a12273fd9ee7a4fa92ed0edde81523d0c56632bf))
* **cuda:** pin libnccl-dev to installed libnccl2 version ([266bcad](https://github.com/tmac1973/llama-toolchest/commit/266bcadfe9a0bd18b31048409341e97f6fdb45c0))
* **tensor-split:** pass --fit off so model load doesn't abort ([4a05a38](https://github.com/tmac1973/llama-toolchest/commit/4a05a389a5b6092f0e9e216ccc00f296be0cd3ab))
* **tensor-split:** pass --fit off so model load doesn't abort ([bf032f6](https://github.com/tmac1973/llama-toolchest/commit/bf032f6523b22ffa2efdd3fceede931d83494850))

## [2.6.0](https://github.com/tmac1973/llama-toolchest/compare/v2.5.0...v2.6.0) (2026-05-25)


### Features

* **cuda:** install NCCL so tensor parallelism uses the optimized AllReduce ([6d30977](https://github.com/tmac1973/llama-toolchest/commit/6d309773669620619dace675afece67c963b5c69))
* **cuda:** install NCCL so tensor parallelism uses the optimized AllReduce ([4e3f2b8](https://github.com/tmac1973/llama-toolchest/commit/4e3f2b81723b0c0485f704a580bcf7ec08f71317))

## [2.5.0](https://github.com/tmac1973/llama-toolchest/compare/v2.4.0...v2.5.0) (2026-05-21)


### Features

* **presets:** scrape HuggingFace sampling parameters and surface as dropdown presets ([fb77608](https://github.com/tmac1973/llama-toolchest/commit/fb776086ffdd9ff53b97bebc70d8046ab40a9bc2))
* **presets:** scrape sampling parameters from HuggingFace and surface as dropdown presets ([79655f1](https://github.com/tmac1973/llama-toolchest/commit/79655f10b64ff1605a2757f41ab66bcac65025b8))

## [2.4.0](https://github.com/tmac1973/llama-toolchest/compare/v2.3.0...v2.4.0) (2026-05-17)


### Features

* **spec:** MTP + draft-resource flags, fix draft mode --spec-type ([a896a75](https://github.com/tmac1973/llama-toolchest/commit/a896a75cdf5bae89d79a07f7ad65c2cc710d28c7))
* **spec:** support MTP, draft-resource flags, and refresh draft mode for current llama.cpp ([b33f7e8](https://github.com/tmac1973/llama-toolchest/commit/b33f7e8daaa8f72abf38a30a427776f6b45046c1))

## [2.3.0](https://github.com/tmac1973/llama-toolchest/compare/v2.2.3...v2.3.0) (2026-05-13)


### Features

* **bench:** edit and re-run batch benchmark jobs ([466875e](https://github.com/tmac1973/llama-toolchest/commit/466875ec34dd4307aba77dfd95e810f3d2c13e39))
* **bench:** edit and re-run batch benchmark jobs ([009cf32](https://github.com/tmac1973/llama-toolchest/commit/009cf3214d2603f76b2c11a4ecd50c838c5f9a67))


### Bug Fixes

* **api:** distinguish live vs. pending restart-requiring fields in /info ([6142957](https://github.com/tmac1973/llama-toolchest/commit/614295780742da6b8f8a6038fd7561c22d7eebdd))
* **api:** report live vs. pending values for restart-requiring config fields ([0b30334](https://github.com/tmac1973/llama-toolchest/commit/0b3033406cc202ecdc2553bdeefaacb460fb6f29))
* **shutdown:** stop child llama-server in parallel with HTTP drain ([08e0f13](https://github.com/tmac1973/llama-toolchest/commit/08e0f1382f0edb7acf1e941bac133a178fbec534))

## [2.2.3](https://github.com/tmac1973/llama-toolchest/compare/v2.2.2...v2.2.3) (2026-05-11)


### Bug Fixes

* **api:** resolve default context_size in /api/models/{id}/info ([f6be200](https://github.com/tmac1973/llama-toolchest/commit/f6be200f33ed1e02fac51f75736a68371c69b3fd))
* **api:** resolve default context_size in model info endpoint ([551d01a](https://github.com/tmac1973/llama-toolchest/commit/551d01acdaca9d605db2877ab8dcc88e69e5b6d1))

## [2.2.2](https://github.com/tmac1973/llama-toolchest/compare/v2.2.1...v2.2.2) (2026-05-11)


### Bug Fixes

* **bench:** show build profile in comparison bar labels ([48c9f10](https://github.com/tmac1973/llama-toolchest/commit/48c9f10933e1757773dfd2ac820895eff0351901))
* **bench:** show build profile in comparison bar labels ([00d73a8](https://github.com/tmac1973/llama-toolchest/commit/00d73a8247def33b04f795c899eaada350dd2993))

## [2.2.1](https://github.com/tmac1973/llama-toolchest/compare/v2.2.0...v2.2.1) (2026-05-11)


### Bug Fixes

* **bench:** preserve cell-check selection across running-job polls ([d7c746b](https://github.com/tmac1973/llama-toolchest/commit/d7c746b78615cf8c572c651de033ff3b53ca9a89))
* **bench:** preserve cell-check selection across running-job polls ([7aa79e2](https://github.com/tmac1973/llama-toolchest/commit/7aa79e27c5ba627670fb81ddca80d4677fe8905a))

## [2.2.0](https://github.com/tmac1973/llama-toolchest/compare/v2.1.2...v2.2.0) (2026-05-10)


### Features

* **hf:** mark downloaded models and gate downloads by free disk space ([3f07aa2](https://github.com/tmac1973/llama-toolchest/commit/3f07aa230711f8740d33c35356f448f5dd42f2a4))
* **hf:** mark downloaded models and gate downloads by free disk space ([01b0af1](https://github.com/tmac1973/llama-toolchest/commit/01b0af181b26b01196a793e6c576c6492a5b6a22))

## [2.1.2](https://github.com/tmac1973/llama-toolchest/compare/v2.1.1...v2.1.2) (2026-05-09)


### Bug Fixes

* **api:** make /v1 and /api/models endpoints behave consistently ([b9519e9](https://github.com/tmac1973/llama-toolchest/commit/b9519e992a83d8d71deebc33d74c774854c227cc))
* **api:** make /v1 and /api/models endpoints behave consistently ([d1cc197](https://github.com/tmac1973/llama-toolchest/commit/d1cc197a671f525535ad6227964a6f573a2a8f58))

## [2.1.1](https://github.com/tmac1973/llama-toolchest/compare/v2.1.0...v2.1.1) (2026-05-09)


### Bug Fixes

* **benchmark:** forward HF_TOKEN and HF_HOME to llama-benchy ([376194c](https://github.com/tmac1973/llama-toolchest/commit/376194ce6c749962260f4c859890133a2421bd46))
* **setup:** install uv in container images for llama-benchy ([ea412fe](https://github.com/tmac1973/llama-toolchest/commit/ea412fe4a2a21c19479bb88e3371a7f2dddab6a2))
* **ui:** unique download progress slot IDs per HF model ([8d2073f](https://github.com/tmac1973/llama-toolchest/commit/8d2073f71330ea0872236686f477206bbe5284a4))
* **ui:** unique download progress slot IDs per HF model ([3f2e414](https://github.com/tmac1973/llama-toolchest/commit/3f2e414d4d3a9f453f671041baa41f26ac1b97b7))

## [2.1.0](https://github.com/tmac1973/llama-toolchest/compare/v2.0.0...v2.1.0) (2026-05-07)


### Features

* **setup:** auto-route up/down/logs and harden host cuda install ([bafbb11](https://github.com/tmac1973/llama-toolchest/commit/bafbb116bce7855f21303e1d3847965cc1199eed))
* **setup:** auto-route up/down/logs and harden host cuda install ([3585dcc](https://github.com/tmac1973/llama-toolchest/commit/3585dcc814dbd2ac03582c087a2a4785da445127))

## [2.0.0](https://github.com/tmac1973/llama-toolchest/compare/v1.9.3...v2.0.0) (2026-05-07)


### ⚠ BREAKING CHANGES

* **ui:** fold the single-run path into a 1-cell job builder (pass 7b)
* **benchmark:** drop llama-bench, rename presets to internal-*

### Features

* **api:** /api/benchmark-jobs endpoints + SSE progress + run filters ([8399d88](https://github.com/tmac1973/llama-toolchest/commit/8399d88c5b19979ff72a5da28e1814b77ded5022))
* **api+ui:** About Benchmarks modal renders live data from server ([952c5e4](https://github.com/tmac1973/llama-toolchest/commit/952c5e4750a9dcafe87b3a2be56e39c0f7ddeb8e))
* **api+ui:** full CSV (cells/summary) + JSON exports for runs and jobs ([874b449](https://github.com/tmac1973/llama-toolchest/commit/874b4492fc0050f0be42ee1f7df3fa6d1d5b69c7))
* **benchmark+ui:** build pre-flight guard, live job updates ([296a35f](https://github.com/tmac1973/llama-toolchest/commit/296a35f44e6ea6812ab3c7fbf060dc85230bb672))
* **benchmark:** drop llama-bench, rename presets to internal-* ([149ad5e](https://github.com/tmac1973/llama-toolchest/commit/149ad5e6ee87c0cbef77392a8b4fb7c934029e76))
* **benchmark:** integrate llama-benchy alongside existing presets ([3652c78](https://github.com/tmac1973/llama-toolchest/commit/3652c788d9121b82745f880876a25d44005a72a5))
* **benchmark:** job model + v2 storage envelope with v1→v2 migration ([4286591](https://github.com/tmac1973/llama-toolchest/commit/4286591235a32e626449b1a6ad8f5f15f262b0a9))
* **benchmark:** JobQueue with sequential per-cell orchestration ([9182c72](https://github.com/tmac1973/llama-toolchest/commit/9182c72e76582c0ec8cae30f9a52164899491095))
* **benchmark:** snapshot the active build on every run ([827570c](https://github.com/tmac1973/llama-toolchest/commit/827570c14addb60cdc70ee3f512990906d6c2504))
* **ui:** fold the single-run path into a 1-cell job builder (pass 7b) ([8e32975](https://github.com/tmac1973/llama-toolchest/commit/8e32975f684d54920cabbbff8db79489f65b0f8e))
* **ui:** jobs list, detail matrix, and new-job form (pass 7a) ([d4b1c15](https://github.com/tmac1973/llama-toolchest/commit/d4b1c15ddacba414156fa4c2b2ad01ad11d6396c))
* **ui:** override fields use dropdowns to match the model config form ([4d4c3e7](https://github.com/tmac1973/llama-toolchest/commit/4d4c3e751d0ee86a642a1926376f788fb562e5d6))
* **ui:** show model quant + preserve open cell-detail across job poll ([75405dc](https://github.com/tmac1973/llama-toolchest/commit/75405dc078e70f10a75c60a2ad6aad9936ea55a1))
* **ui:** wider model column, tooltips, build column, client-side sort ([70dbcbc](https://github.com/tmac1973/llama-toolchest/commit/70dbcbc094283b93d17ea08bea3180efe369cb8a))


### Bug Fixes

* **benchmark:** include sentencepiece + tiktoken in uvx environment ([113b081](https://github.com/tmac1973/llama-toolchest/commit/113b081f7d587cd3f96052d9fdfcaac3fe884a82))
* **setup:** host install summary reports the configured port ([43ef7e2](https://github.com/tmac1973/llama-toolchest/commit/43ef7e21e19b98555532efcf29928e91b9fc83b1))
* **ui:** drop list polling, push live updates via OOB swaps from detail ([319d2b6](https://github.com/tmac1973/llama-toolchest/commit/319d2b6c9926e2495676f00aee1812305573b4c2))
* **ui:** pivot compare details table — one row per run ([a2d7d94](https://github.com/tmac1973/llama-toolchest/commit/a2d7d947ecb00c24021af84067b8364b91ca3179))
* **ui:** re-fetch detail directly after list refresh, not via event trigger ([b9897c6](https://github.com/tmac1973/llama-toolchest/commit/b9897c6380e13d43019161f27ee6bfe37c84aa66))
* **ui:** remove redundant Benchmark Results section, scope bulk actions ([356a9d4](https://github.com/tmac1973/llama-toolchest/commit/356a9d4f501037a2c5d37ed50b18feffc24533b5))
* **ui:** show "f16" for default KV cache quant instead of dash/hidden ([4f8dc63](https://github.com/tmac1973/llama-toolchest/commit/4f8dc63ba2b4092293f0d527335bd2c9b1be38e3))
* **ui:** use htmx.ajax for compare swap so inline &lt;script&gt; runs ([45756cb](https://github.com/tmac1973/llama-toolchest/commit/45756cb77c5716e837aa48a1986897b219eef4f4))

## [1.9.3](https://github.com/tmac1973/llama-toolchest/compare/v1.9.2...v1.9.3) (2026-05-06)


### Bug Fixes

* **server:** restart picks up the latest active build, settings persist ([27aaa87](https://github.com/tmac1973/llama-toolchest/commit/27aaa870aefbee45054e186d192591bd1017c2ce))

## [1.9.2](https://github.com/tmac1973/llama-toolchest/compare/v1.9.1...v1.9.2) (2026-05-06)


### Bug Fixes

* **vram:** treat ctx_size=0 as the model's trained context, not 2048 ([d079a5e](https://github.com/tmac1973/llama-toolchest/commit/d079a5ed743d9194867bba6bdbad41c2e994c23c))

## [1.9.1](https://github.com/tmac1973/llama-toolchest/compare/v1.9.0...v1.9.1) (2026-05-06)


### Bug Fixes

* **config:** stop env-var leak that lost downloaded models ([ffbe58e](https://github.com/tmac1973/llama-toolchest/commit/ffbe58e8fd3c9aaf89bab7dc8d9be8c19174eff8))

## [1.9.0](https://github.com/tmac1973/llama-toolchest/compare/v1.8.0...v1.9.0) (2026-05-06)


### Features

* **models:** per-model parallel request slots ([6c25ebd](https://github.com/tmac1973/llama-toolchest/commit/6c25ebd5c4181cb7b6cf5006d53951384f52acb9))

## [1.8.0](https://github.com/tmac1973/llama-toolchest/compare/v1.7.1...v1.8.0) (2026-05-05)


### Features

* **browse:** add HuggingFace link icon to search results ([f686085](https://github.com/tmac1973/llama-toolchest/commit/f686085581311aef3b14f9f4fa970abf5373faf5))
* **models:** add HuggingFace link icon next to model name ([9d462fc](https://github.com/tmac1973/llama-toolchest/commit/9d462fc2d877dc143914981c74e89f98494b2956))

## [1.7.1](https://github.com/tmac1973/llama-toolchest/compare/v1.7.0...v1.7.1) (2026-05-05)


### Bug Fixes

* **benchmarks:** reattach progress UI when returning to the tab ([3df3f4f](https://github.com/tmac1973/llama-toolchest/commit/3df3f4fec28cc88143b8a7542c219c193f6840de))
* **monitor:** strip card/GPU prefix from rocm-smi device field ([0e701e7](https://github.com/tmac1973/llama-toolchest/commit/0e701e796f41f068a8cf72b442ea1729589ce77b))

## [1.7.0](https://github.com/tmac1973/llama-toolchest/compare/v1.6.0...v1.7.0) (2026-05-05)


### Features

* add deps command and SDK section in status --host ([f7ec150](https://github.com/tmac1973/llama-toolchest/commit/f7ec1505e3baf63d02099d4bada89a6dfda8cf1d))
* add migrate command for switching between container and host installs ([c090de2](https://github.com/tmac1973/llama-toolchest/commit/c090de2771c4a33c5df3a049219e839dbbcf5c87))
* **host:** support multi-backend SDK installs (--cuda/--rocm/--vulkan) ([c8d2a76](https://github.com/tmac1973/llama-toolchest/commit/c8d2a766697a762aff508064ef89bee12a6f7f6a))


### Bug Fixes

* emit mode-specific speculative decoding flags ([b769934](https://github.com/tmac1973/llama-toolchest/commit/b769934b7a453715494faf403bc971abf72bccb1))
* **host:** use glslc package on Debian instead of glslang-tools ([ab2d148](https://github.com/tmac1973/llama-toolchest/commit/ab2d148ace6b1cee4058c4f3ab6107080b598c99))
* **migrate:** translate mmproj_path/draft_model_path across the boundary ([a19eade](https://github.com/tmac1973/llama-toolchest/commit/a19eade8c11f3a9298234d07a3308309a7174033))

## [1.6.0](https://github.com/tmac1973/llama-toolchest/compare/v1.5.1...v1.6.0) (2026-05-05)


### Features

* redesign models tab as flat list of cards ([d82921b](https://github.com/tmac1973/llama-toolchest/commit/d82921ba0ebab7d24460da557872e888612eb8cb))
* restrict vulkan builds to host mode and expand vulkan deps ([b821347](https://github.com/tmac1973/llama-toolchest/commit/b82134715264e7f45488c767f62a6b1b76dfb83f))

## [1.5.1](https://github.com/tmac1973/llama-toolchest/compare/v1.5.0...v1.5.1) (2026-05-04)


### Bug Fixes

* auto-refresh model and build lists after state changes ([e7e9859](https://github.com/tmac1973/llama-toolchest/commit/e7e9859daa61d3f0d00980041c3b8d361e30f2b5))

## [1.5.0](https://github.com/tmac1973/llama-toolchest/compare/v1.4.2...v1.5.0) (2026-05-01)


### Features

* GPU allocation map uses peak VRAM (weights + KV cache) ([728c8e5](https://github.com/tmac1973/llama-toolchest/commit/728c8e5131bdf3caaea964a7d64c0b019ce56464))

## [1.4.2](https://github.com/tmac1973/llama-toolchest/compare/v1.4.1...v1.4.2) (2026-05-01)


### Bug Fixes

* server log Clear button now drops the buffered history ([280d3b4](https://github.com/tmac1973/llama-toolchest/commit/280d3b49ff57efedd32c001a509d398be13427b5))

## [1.4.1](https://github.com/tmac1973/llama-toolchest/compare/v1.4.0...v1.4.1) (2026-05-01)


### Bug Fixes

* dashboard "restart needed" false-positive and Available Models layout ([47bccff](https://github.com/tmac1973/llama-toolchest/commit/47bccffa78c4b76e33534580308cc00b163967b4))

## [1.4.0](https://github.com/tmac1973/llama-toolchest/compare/v1.3.0...v1.4.0) (2026-04-30)


### Features

* setup.sh quick now upgrades the package without rebuilding the image ([b83e6f2](https://github.com/tmac1973/llama-toolchest/commit/b83e6f2cf6a9debeb8c99f288bc965b509a65c71))

## [1.3.0](https://github.com/tmac1973/llama-toolchest/compare/v1.2.1...v1.3.0) (2026-04-30)


### Features

* show version in sidebar under "Inference Manager" ([e3b1235](https://github.com/tmac1973/llama-toolchest/commit/e3b1235739661d76a70c0ed95929c5a42ce223e7))


### Bug Fixes

* model config: don't reset speculative decoding fields on every save ([795e2b2](https://github.com/tmac1973/llama-toolchest/commit/795e2b2acf4b8b46cd0570a68fc40363f4b1a824))

## [1.2.1](https://github.com/tmac1973/llama-toolchest/compare/v1.2.0...v1.2.1) (2026-04-30)


### Bug Fixes

* don't warn about port conflicts caused by our own container ([2f71f95](https://github.com/tmac1973/llama-toolchest/commit/2f71f958e719d8e0f239b5ae66b1f3261abaeae1))
* equalize Info/Delete button heights on Builds page ([9f0ab74](https://github.com/tmac1973/llama-toolchest/commit/9f0ab74d8c858bc6d9faa7b8f86440ffbffb779b))
* migrate_legacy_volume removes the pre-rename container too ([cbea639](https://github.com/tmac1973/llama-toolchest/commit/cbea639349800a699fd2cbcc597ba7aaab4d41cc))
* portable container-existence check (Docker compatibility) ([b3327d0](https://github.com/tmac1973/llama-toolchest/commit/b3327d005c0f85da562a331239d10ca6b68591dc))
* silence Compose warnings on migrated install ([5bf3fe6](https://github.com/tmac1973/llama-toolchest/commit/5bf3fe652ec09f0a0d6cb6dfcf0d66c25c03156a))

## [1.2.0](https://github.com/tmac1973/llama-toolchest/compare/v1.1.0...v1.2.0) (2026-04-30)


### Features

* editable models directory in Settings ([05a2637](https://github.com/tmac1973/llama-toolchest/commit/05a2637f3a3127624532fc244e020065e5416363))


### Bug Fixes

* detect distro family before host install dispatch ([ded3367](https://github.com/tmac1973/llama-toolchest/commit/ded3367550051bf5047a0f27dda9994dbafdcd5f))
* drop unused openblas Recommends from package ([c7e8675](https://github.com/tmac1973/llama-toolchest/commit/c7e8675244c1057bb327fcbdcca7ba692acbf61a))

## [1.1.0](https://github.com/tmac1973/llama-toolchest/compare/v1.0.0...v1.1.0) (2026-04-29)


### Features

* --host install now defaults to fetching released .deb/.rpm ([49c40f6](https://github.com/tmac1973/llama-toolchest/commit/49c40f658a33caa09361f9eacc2fd4633b8d72e5))
* Dockerfiles default to installing released package ([21572e2](https://github.com/tmac1973/llama-toolchest/commit/21572e2765ae9cf6247dfcbea00ad1c07b8558c0))

## 1.0.0 (2026-04-29)


### ⚠ BREAKING CHANGES

* Existing container deployments must run ./setup.sh rebuild to migrate the llamactl-data volume to llama-toolchest-data; .env files using LLAMACTL_* are auto-rewritten to LLAMA_TOOLCHEST_*.

### Features

* containerless host install and rename to llama-toolchest ([52e5c46](https://github.com/tmac1973/llama-toolchest/commit/52e5c46f238d89ab8019ba209845ea9474daa7f2))
