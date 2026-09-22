# Phase 06 — Documentation

**Depends on:** Phases 03, 05 · **Enables:** nothing — this is the last phase

## Goal

Make the option discoverable and honestly described in `README.md`: what it is,
why it is container-only, how to pin a version, what the host kernel needs to be,
and that switching ROCm versions means rebuilding `llama.cpp`. The compatibility
table currently states a single ROCm version for AMD, which becomes wrong the
moment Phase 03 lands, so this phase is not optional polish. It depends on Phase
05 because the text must describe what was actually observed — including any
caveat or failure — rather than what was hoped for.

## Files touched

- `README.md` — the compatibility table (around `:114`) and the ROCm section
  (around `:180`); one line in the `setup.sh` flag description (around `:55`) and
  the subcommand list (around `:99`).

## Steps

1. **Compatibility table** (`README.md:114`). The AMD row reads
   `| AMD | ROCm 7.2 | rocm, cpu, vulkan† | RDNA and newer. |`. Set the version
   cell to exactly `ROCm 7.2 (default), ROCm 10 ‡` and leave the other three
   cells unchanged. Add a `‡` footnote below the table, in the style of the
   existing `†` note, reading: ROCm 10 is available in container mode only,
   because AMD publishes it only as a container image — see the ROCm section
   below.
2. **ROCm section** (`README.md:180`). Keep the existing
   `HSA_OVERRIDE_GFX_VERSION` paragraph and add a short subsection covering, in
   this order:
   - **What it is.** An experimental ROCm container built on AMD's own
     `rocm/dev-ubuntu-24.04` image instead of Fedora plus RPMs.
   - **Why it exists.** ROCm 10 and the 7.14.x line are published only as
     container images; `repo.radeon.com`'s `el9`, `el10`, `rhel9` and `rhel10`
     paths all stop at 7.2.4. State this plainly — it is the single fact that
     explains the whole design, and a reader who does not know it will assume
     the default is simply out of date.
   - **How to choose it.** The install prompt, and the non-interactive forms:
     ```
     ./setup.sh install --rocm-next                  # experimental, default ROCm release
     ./setup.sh install --rocm-image 10.0.0-full     # experimental, pinned
     ```
     Mention `ROCM_VARIANT` and `ROCM_BASE_IMAGE` as the environment equivalents,
     that the choice is stored in `.env` so `rebuild` keeps it, and that any tag
     from the AMD image repository can be given (link
     `https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags`). Name the tags
     confirmed working in Phase 05 — `10.0.0-full` and `7.14.1-full` if both
     passed there.
   - **What it requires.** That ROCm 10 supports RDNA 1 and newer and the CDNA
     cards — it does *not* require RDNA 4 — and that the host kernel needed
     depends on the card, not on ROCm: RDNA 4 wants 6.12, RDNA 3 6.0, RDNA 2
     5.9, RDNA 1 5.3. Say that setup.sh checks the detected card against both
     lists and warns without blocking, and that the requirement is on the host
     driver, because the container carries no kernel components.
   - **Switching.** `llama.cpp` builds are compiled against one ROCm version and
     will not load under another, so switching means rebuilding from the Builds
     page. Say that nothing is deleted, so switching back makes the old builds
     work again, and that the Builds page flags builds that do not match the
     running container. Cross-reference the existing note at `README.md:76`,
     which already makes the same point for host-to-container migration, so the
     two read as one rule rather than two coincidences.
   - **Status.** That it is experimental: ROCm 10 is not in AMD's validated
     llama.cpp support matrix, which lists 7.0.0, 6.4.3, 6.4.2 and 6.4.1. Add
     whatever Phase 05 found: the throughput comparison from its Step 8, which
     is a required measurement there rather than an optional one, and any step
     that failed. If verification failed, say so here; the overview requires the
     failure be documented rather than hidden.
   - **Host installs.** One sentence: `--rocm` host installs use
     `repo.radeon.com` and cannot install ROCm 10, because there are no packages
     for it there.
3. **Flag description** (`README.md:55`). That paragraph explains `--cuda`,
   `--rocm` and `--vulkan` and that each implies `--host`. Add a sentence
   distinguishing the new flags: `--rocm-next` and `--rocm-image` are
   container-only and do **not** imply `--host`. The similar names invite exactly
   that confusion, so it is worth the sentence.
4. **Subcommand list** (`README.md:99`). The `detect` line says it prints the
   detected GPU backend; note that for AMD it also prints the ROCm variant that
   would be used.
5. **Read the whole ROCm section once more** for the plain-language standard the
   rest of the README holds to: no unexplained jargon, and every requirement
   stated as something the reader can check on their own machine (a kernel
   version they can read with `uname -r`, a tag they can look up).

## Build gate

```
grep -n "ROCm 10" README.md          # the table, footnote and section all mention it
grep -n "rocm-image" README.md       # the flag is documented
```

No compiler or test involvement — this phase changes only Markdown. Re-run
`go build ./... && go test ./...` once at the end to confirm the branch as a
whole is still green before opening the pull request.

## Test plan

1. **A reader can get there from the table.** Start at the compatibility table,
   follow the footnote, and confirm the ROCm section answers what the option is,
   how to select it and what it requires, without needing the plan folder.
2. **Every command in the docs runs.** Copy each command out of the new section
   and run it: `--rocm-next` and `--rocm-image 10.0.0-full` both reach the
   expected variant via `./setup.sh detect`.
3. **Every claim is one Phase 05 actually observed.** Walk the Status paragraph
   against `findings-verification.md` line by line and delete any claim not
   recorded there. Deleting rather than labelling: an "untested" note in a
   compatibility section reads as a hedge and invites the reader to try it
   anyway, and a claim worth keeping is worth verifying in Phase 05 instead. The
   throughput figures must match its Step 8 exactly.
4. **Links resolve.** The Docker Hub tags link loads and shows the tag named as
   the default.
5. **No stale version claims.** `grep -n "ROCm 7.2" README.md` — every remaining
   mention is still accurate now that two versions are possible, in particular
   the `HSA_OVERRIDE_GFX_VERSION` paragraph, which is about older GPUs and
   applies to both variants.

## Commit

```
docs: describe the experimental ROCm container option

The compatibility table claimed a single ROCm version for AMD, which
stopped being true when the variant selector landed. Documents what the
experimental container is, why ROCm 10 can only come from a container
image, how to pin a version, the host kernel it needs, and that
switching ROCm versions means rebuilding llama.cpp.
```

## Rollback

Revert `README.md`. Documentation only — nothing depends on it and reverting
cannot affect a running install. Safe to leave partially applied, though a
half-updated compatibility table is worse than either state: if this phase is
abandoned, revert the table row rather than leaving it naming a version the
script no longer guarantees.
