# scripts/lib/host.sh
#
# Host install logic — installs llama-toolchest directly on the host system
# rather than inside a container. Sourced by setup.sh; depends on common
# helpers (log/ok/warn/err/prompt_confirm) and on scripts/lib/service.sh.
#
# Strategy: build the binary from source via `go build`. Once a tagged
# release exists on GitHub, this can switch to fetching the prebuilt
# package; until then, from-source is the only working path.
#
# Layout (Linux):
#   user install (default)
#     binary:   ~/.local/bin/llama-toolchest
#     config:   $XDG_CONFIG_HOME/llama-toolchest/llama-toolchest.yaml
#     data:     $XDG_DATA_HOME/llama-toolchest
#     unit:     ~/.config/systemd/user/llama-toolchest.service
#   system install (root)
#     binary:   /usr/local/bin/llama-toolchest
#     config:   /etc/llama-toolchest/llama-toolchest.yaml
#     data:     /var/lib/llama-toolchest
#     unit:     /etc/systemd/system/llama-toolchest.service
#
# Public functions:
#   host_install
#   host_uninstall
#   host_status
#   host_is_installed
#   host_up
#   host_down
#   host_logs

host_scope() {
    if [[ $EUID -eq 0 ]]; then
        echo "system"
    else
        echo "user"
    fi
}

host_bin_dir() {
    if [[ "$(host_scope)" == "system" ]]; then
        echo "/usr/local/bin"
    else
        echo "${HOME}/.local/bin"
    fi
}

host_config_dir() {
    if [[ "$(host_scope)" == "system" ]]; then
        echo "/etc/llama-toolchest"
    else
        echo "${XDG_CONFIG_HOME:-$HOME/.config}/llama-toolchest"
    fi
}

host_data_dir() {
    if [[ "$(host_scope)" == "system" ]]; then
        echo "/var/lib/llama-toolchest"
    else
        echo "${XDG_DATA_HOME:-$HOME/.local/share}/llama-toolchest"
    fi
}

host_binary_path() {
    echo "$(host_bin_dir)/llama-toolchest"
}

host_config_path() {
    echo "$(host_config_dir)/llama-toolchest.yaml"
}

# Verify that the prerequisites for an in-tree `go build` are present. Does
# not install anything — leaves that to the user, since the build toolchain
# differs by distro and the existing setup.sh prereq logic already handles
# it for the container path.
host_check_build_toolchain() {
    local missing=()
    command -v go      >/dev/null 2>&1 || missing+=("go (>= 1.25)")
    command -v cmake   >/dev/null 2>&1 || missing+=("cmake")
    command -v ninja   >/dev/null 2>&1 || command -v ninja-build >/dev/null 2>&1 || missing+=("ninja-build")
    command -v git     >/dev/null 2>&1 || missing+=("git")

    if [[ ${#missing[@]} -gt 0 ]]; then
        warn "Missing build prerequisites: ${missing[*]}"
        log "These are needed to build the llama-toolchest binary and (later) llama.cpp from the UI."
        log "Install via your package manager:"
        echo "    Debian/Ubuntu:  sudo apt-get install golang cmake ninja-build git build-essential"
        echo "    Fedora:         sudo dnf install golang cmake ninja-build git gcc-c++ make"
        echo "    Arch:           sudo pacman -S go cmake ninja git base-devel"
        return 1
    fi
    return 0
}

# Locate a side-by-side g++ that nvcc will accept as host compiler. The
# default system g++ on modern distros (gcc 16 on Fedora 44) outpaces what
# CUDA's host_config.h will allow, so the builder explicitly hands nvcc a
# compat compiler when one is installed. Echoes the absolute path on
# success (returns 0); silent on miss (returns 1).
host_find_cuda_host_compiler() {
    for c in /usr/bin/g++-15 /usr/bin/g++-14 /usr/bin/g++-13 /usr/bin/g++-12; do
        if [[ -x "$c" ]]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

# Locate nvcc, looking past PATH. NVIDIA's official RPM/DEB installs it
# under /usr/local/cuda/bin without touching PATH, so a plain `command -v`
# misses it. Echoes the absolute path if found and returns 0; otherwise
# returns 1 with no output. Used by both the SDK-presence check and the
# builder PATH augmentation guidance.
host_find_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return 0
    fi
    for candidate in /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# Locate a ROCm tool, looking past PATH. ROCm installs under /opt/rocm (or
# $ROCM_PATH) and only some packagings drop symlinks into /usr/bin, so a
# plain `command -v` misses a perfectly good install. Mirrors FindROCmTool
# in internal/builder/detect.go so setup.sh and the builder agree on what
# counts as "ROCm is here". Echoes the absolute path (returns 0); silent
# on miss (returns 1).
host_find_rocm_tool() {
    local name="$1"
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return 0
    fi
    local -a dirs=()
    [[ -n "${ROCM_PATH:-}" ]] && dirs+=("${ROCM_PATH}/bin")
    dirs+=(/opt/rocm/bin)
    local d
    for d in "${dirs[@]}"; do
        if [[ -x "$d/$name" ]]; then
            echo "$d/$name"
            return 0
        fi
    done
    return 1
}

# Echo the ROCm install prefix. There is no single answer: AMD's own
# packages land in /opt/rocm, Fedora's native RPMs install into /usr with
# no /opt/rocm at all, and source builds go wherever they were configured.
# Ask hipconfig — it reports the prefix of the install it belongs to —
# before falling back to guesses. Returns 1 if no prefix is identifiable.
host_rocm_prefix() {
    if [[ -n "${ROCM_PATH:-}" && -d "${ROCM_PATH}" ]]; then
        echo "${ROCM_PATH}"
        return 0
    fi
    local hipconfig prefix
    if hipconfig="$(host_find_rocm_tool hipconfig)"; then
        prefix="$("$hipconfig" --rocmpath 2>/dev/null)"
        if [[ -n "$prefix" && -d "$prefix" ]]; then
            echo "$prefix"
            return 0
        fi
    fi
    if [[ -d /opt/rocm ]]; then
        echo /opt/rocm
        return 0
    fi
    local hipcc
    if hipcc="$(host_find_rocm_tool hipcc)"; then
        prefix="$(dirname "$(dirname "$hipcc")")"
        if [[ -d "$prefix" ]]; then
            echo "$prefix"
            return 0
        fi
    fi
    return 1
}

# Echo every prefix that could hold a ROCm install, one per line. There
# is no single answer and the consumers disagree: llama.cpp's
# ggml-hip/CMakeLists.txt takes $ROCM_PATH, else /opt/rocm, else /usr;
# the builder (internal/builder/builder.go) derives ROCM_PATH from the
# directory holding hipconfig; and Debian/Ubuntu register /opt/rocm
# through update-alternatives while the files themselves live under
# /usr. Checking only one prefix is how a host with a complete toolchain
# gets told its SDK is missing — probe all of them.
host_rocm_prefix_candidates() {
    local -a out=()
    local p tool
    [[ -n "${ROCM_PATH:-}" ]] && out+=("${ROCM_PATH}")
    if p="$(host_rocm_prefix 2>/dev/null)"; then
        out+=("$p")
    fi
    for tool in hipconfig hipcc; do
        if p="$(host_find_rocm_tool "$tool")"; then
            out+=("$(dirname "$(dirname "$p")")")
        fi
    done
    out+=(/opt/rocm /usr)
    printf '%s\n' "${out[@]}" | awk 'NF && !seen[$0]++'
}

# Whether one ROCm cmake config package is on disk somewhere cmake will
# look. Three library layouts are in play and they are not
# interchangeable: AMD's own (<prefix>/lib/cmake/<pkg>), Fedora's
# (<prefix>/lib64/cmake/<pkg>) and Debian multiarch
# (<prefix>/lib/x86_64-linux-gnu/cmake/<pkg>) — the last is the one
# CMakeDetermineHIPCompiler reports in its "expected at one of" error.
host_rocm_have_cmake_pkg() {
    local pkg="$1" prefix
    while read -r prefix; do
        [[ -n "$prefix" ]] || continue
        if compgen -G "${prefix}/lib/cmake/${pkg}/${pkg}-config.cmake" >/dev/null 2>&1; then
            return 0
        fi
        if compgen -G "${prefix}/lib64/cmake/${pkg}/${pkg}-config.cmake" >/dev/null 2>&1; then
            return 0
        fi
        if compgen -G "${prefix}/lib/*/cmake/${pkg}/${pkg}-config.cmake" >/dev/null 2>&1; then
            return 0
        fi
    done < <(host_rocm_prefix_candidates)
    return 1
}

# Echo the clang++ that can compile HIP device code, or nothing. Mirrors
# findHIPCompiler in internal/builder/builder.go so setup.sh and the
# builder agree on what counts as a usable HIP compiler. Ask hipconfig
# first (a ROCm that ships its own llvm knows best), then the usual ROCm
# roots, then a distro clang — but only one with the AMD device bitcode
# beside it, since Debian/Ubuntu package those per LLVM version and the
# newest clang on the box is regularly the wrong one.
host_find_hip_clang() {
    local hipconfig dir root
    if hipconfig="$(host_find_rocm_tool hipconfig)"; then
        dir="$("$hipconfig" --hipclangpath 2>/dev/null)" || dir=""
        if [[ -n "$dir" && -x "${dir}/clang++" ]]; then
            echo "${dir}/clang++"
            return 0
        fi
    fi
    for root in "${ROCM_PATH:-}" /opt/rocm /usr/lib/rocm /usr/lib64/rocm; do
        [[ -n "$root" ]] || continue
        if [[ -x "${root}/llvm/bin/clang++" ]]; then
            echo "${root}/llvm/bin/clang++"
            return 0
        fi
    done
    local best="" best_ver=-1 candidate ver
    for candidate in /usr/lib/llvm-*/bin/clang++; do
        [[ -x "$candidate" ]] || continue
        root="$(dirname "$(dirname "$candidate")")"
        compgen -G "${root}/lib/clang/*/amdgcn/bitcode" >/dev/null 2>&1 || continue
        ver="${root##*/llvm-}"
        [[ "$ver" =~ ^[0-9]+$ ]] || continue
        if (( ver > best_ver )); then
            best="$candidate"
            best_ver="$ver"
        fi
    done
    [[ -n "$best" ]] || return 1
    echo "$best"
}

# llama.cpp's HIP backend hard-fails below this — see the version gate in
# ggml/src/ggml-hip/CMakeLists.txt ("At least ROCM/HIP V6.1 is required").
# Worth checking up front: Ubuntu 24.04 packages ROCm 5.7, so a host can
# have every dev package installed, a working HIP compiler, and still be
# unable to build.
HOST_ROCM_MIN_VERSION="6.1"

# Echo the installed ROCm version as major.minor (e.g. "6.4"), or nothing
# if it can't be determined.
host_rocm_version() {
    local hipconfig raw
    hipconfig="$(host_find_rocm_tool hipconfig)" || return 1
    raw="$("$hipconfig" --version 2>/dev/null)" || return 1
    [[ "$raw" =~ ([0-9]+)\.([0-9]+) ]] || return 1
    echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
}

# The cmake config packages llama.cpp's HIP backend needs, in the order
# it asks for them. hip-lang is first because enable_language(HIP) — the
# very first thing ggml-hip does — fails on its absence, and it is the
# one a "hipcc is installed, so ROCm is fine" check misses: on
# Debian/Ubuntu hipcc and hip-lang-config.cmake ship in different
# packages (hipcc vs libamdhip64-dev).
HOST_ROCM_CMAKE_PKGS=(hip-lang hip rocblas hipblas)

# Echo the cmake config packages that are absent; empty when the host can
# compile the HIP backend.
host_rocm_missing_cmake_pkgs() {
    local pkg
    local -a absent=()
    for pkg in "${HOST_ROCM_CMAKE_PKGS[@]}"; do
        host_rocm_have_cmake_pkg "$pkg" || absent+=("$pkg")
    done
    echo "${absent[*]}"
}

# Whether the host already has everything llama.cpp's HIP backend needs to
# compile, regardless of how it got there. Package names for ROCm differ
# between AMD's own repo, Debian/Ubuntu's native packaging, and source or
# TheRock installs, so name-based detection reports false "missing" on
# hosts that build fine. Probe capabilities instead: the hipcc compiler
# plus the cmake config packages that llama.cpp's find_package() calls
# resolve. rocwmma and rccl are deliberately not required — they are
# optional accelerations, not build blockers.
host_rocm_sdk_usable() {
    host_find_rocm_tool hipcc >/dev/null 2>&1 || return 1
    [[ -z "$(host_rocm_missing_cmake_pkgs)" ]]
}

# Whether an apt package name resolves to something installable. Used to
# choose between naming schemes rather than emitting a name apt will
# reject. A package present in the index but with no candidate version
# (Candidate: (none)) does not count.
#
# Deliberately not a pipeline: under `set -o pipefail`, `apt-cache policy
# | grep -q` reports failure whenever grep short-circuits on the match
# and apt-cache dies of SIGPIPE, or whenever apt-cache exits non-zero
# over an unrelated sources.list complaint. Either one makes every
# package look unavailable, which is how a host that carries the whole
# ROCm stack in `universe` ends up being handed AMD's package names.
host_apt_pkg_available() {
    local policy
    policy="$(LC_ALL=C apt-cache policy "$1" 2>/dev/null)" || true
    grep -q '^[[:space:]]*Candidate:[[:space:]]*[^([:space:]]' <<<"$policy"
}

# Whether a package is installed, as opposed to merely known to dpkg.
# `dpkg -s` exits 0 for a removed-but-not-purged package (Status:
# deinstall ok config-files), which would have us skip installing
# something that isn't there.
host_dpkg_installed() {
    [[ "$(dpkg-query -W -f='${db:Status-Status}' "$1" 2>/dev/null)" == "installed" ]]
}

# Pick the apt package name to use for one ROCm component. Candidates are
# passed AMD-repo name first, Debian/Ubuntu-native name second (AMD ships
# rocblas-dev, Debian ships librocblas-dev for the same thing).
#   returns 0, echoes nothing  — one of the candidates is already installed
#   returns 0, echoes a name   — install this one
#   returns 1                  — no configured repo carries any candidate
# Callers decide what to do with the last case; emitting an unresolvable
# name is never right when a resolvable one exists, which is the bug this
# replaces.
host_apt_rocm_pkg() {
    local pkg
    for pkg in "$@"; do
        if host_dpkg_installed "$pkg"; then
            return 0
        fi
    done
    for pkg in "$@"; do
        if host_apt_pkg_available "$pkg"; then
            echo "$pkg"
            return 0
        fi
    done
    return 1
}

# Whether AMD's ROCm apt repository is configured. Match the /rocm path
# specifically, not just the repo.radeon.com host: `amdgpu-install` adds
# repo.radeon.com/amdgpu (kernel driver) and /graphics without adding the
# ROCm package repo, which is exactly how a host ends up reporting a live
# ROCm runtime while apt can't resolve a single ROCm dev package. Treating
# the driver repo as "ROCm is configured" would skip the one fix that
# helps. Immune to how the sources file was named.
host_rocm_apt_repo_configured() {
    grep -rqs 'repo\.radeon\.com/rocm' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null
}

# Whether to prefer AMD's package names over the distro's. Both name the
# same libraries but install different ROCm versions, and mixing them is
# worse than either alone — AMD's land under /opt/rocm, the distro's under
# /usr, and a build that picks up 6.x headers against a 7.x runtime fails
# in ways that look like llama.cpp bugs. So: complete whichever install is
# already here. An existing /opt/rocm or a configured ROCm repo means AMD;
# anything else means the distro's own packaging.
host_rocm_prefer_amd_packages() {
    host_rocm_apt_repo_configured && return 0
    # An /opt/rocm made of symlinks into /usr is Debian/Ubuntu's own
    # packaging, not AMD's: the distro registers /opt/rocm through
    # update-alternatives so software that hardcodes the path still
    # works, while the files live in /usr. Treating that as "AMD is
    # installed here" picks AMD's package names on a host whose repos
    # only carry the distro's — every name then fails to resolve and
    # apt-get aborts without installing anything (issue #142).
    if [[ -d /opt/rocm ]]; then
        local libdir
        libdir="$(readlink -f /opt/rocm/lib 2>/dev/null || true)"
        [[ "$libdir" == /usr/* ]] && return 1
        return 0
    fi
    return 1
}

# Register AMD's ROCm apt repository. Debian/Ubuntu now ship ROCm natively
# in universe, but that lags AMD's releases by a lot, so users on new
# hardware need AMD's repo to get a version their GPU supports. Follows
# AMD's documented registration (keyring + sources entry + pin), using the
# "latest" channel so we do not hardcode a ROCm version that ages out.
# Returns 1 if we could not reach the repo or the release has no build for
# this distro codename — caller continues with whatever apt already has.
host_install_amd_rocm_repo_debian() {
    if host_rocm_apt_repo_configured; then
        return 0
    fi

    local codename=""
    if [[ -r /etc/os-release ]]; then
        # UBUNTU_CODENAME is what derivatives (Mint, Pop, Zorin) set to the
        # upstream suite; VERSION_CODENAME on those names the derivative
        # release, which AMD does not publish for.
        codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}")"
    fi
    if [[ -z "$codename" ]]; then
        warn "Couldn't determine the distro codename — skipping AMD ROCm repo setup."
        return 1
    fi

    log "AMD's ROCm apt repository is not configured."
    log "Without it, apt only sees the ROCm version your distro packages, which"
    log "may be too old for a recent GPU."
    if ! prompt_confirm "Add AMD's ROCm repository now? (repo.radeon.com, codename ${codename})"; then
        warn "Skipped — continuing with the packages apt already knows about."
        return 1
    fi

    local rocm_url="https://repo.radeon.com/rocm/apt/latest"
    if ! curl -fsI "${rocm_url}/dists/${codename}/Release" >/dev/null 2>&1; then
        warn "AMD publishes no ROCm build for '${codename}' at ${rocm_url}."
        log "Pick a supported release from:"
        echo "    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html"
        return 1
    fi

    run_sudo install -d -m 0755 /etc/apt/keyrings || return 1
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
        | gpg --dearmor \
        | run_sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null || return 1
    run_sudo chmod 0644 /etc/apt/keyrings/rocm.gpg || return 1

    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] ${rocm_url} ${codename} main" \
        | run_sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null || return 1

    # AMD's documented pin. Their packages and the distro's native ROCm
    # collide on file paths under /opt/rocm; without the pin apt can mix
    # the two into an install that does not link.
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
        | run_sudo tee /etc/apt/preferences.d/rocm-pin-600 >/dev/null || return 1

    run_sudo apt-get update -qq || return 1
    ok "AMD ROCm repository configured (${codename})"
    return 0
}

# Echo the apt candidate names for an optional component, in the order
# host_rocm_prefer_amd_packages dictates.
host_rocm_optional_names() {
    local amd distro
    case "$1" in
        rocwmma) amd="rocwmma-dev"; distro="librocwmma-dev" ;;
        rccl)    amd="rccl-dev";    distro="librccl-dev" ;;
        *)       return 1 ;;
    esac
    if host_rocm_prefer_amd_packages; then
        echo "$amd $distro"
    else
        echo "$distro $amd"
    fi
}

# Echo the human names of the optional ROCm extras whose headers aren't on
# disk ("rocWMMA RCCL"); empty if both are present or ROCm isn't installed.
host_rocm_optional_absent() {
    local prefix
    prefix="$(host_rocm_prefix)" || return 0
    local -a absent=()
    [[ -e "${prefix}/include/rocwmma/rocwmma.hpp" ]] || absent+=("rocWMMA")
    if [[ ! -e "${prefix}/include/rccl/rccl.h" && ! -e "${prefix}/include/rccl.h" ]]; then
        absent+=("RCCL")
    fi
    echo "${absent[*]}"
}

# rocWMMA and RCCL back two opt-in toggles in the ROCm build profile
# (GGML_HIP_ROCWMMA_FATTN, GGML_HIP_RCCL — see internal/builder/profiles.go).
# Neither blocks a build, so neither belongs in the missing-packages list —
# listing them there is what turned an optional accelerant into a hard
# "SDK missing" verdict. Offer them separately, clearly marked optional.
host_offer_rocm_optional() {
    [[ "$1" == "rocm" ]] || return 0
    # Only worth raising once the required set is in place — otherwise the
    # user has a bigger problem than an opt-in toggle.
    [[ -z "$(host_missing_gpu_sdk_packages rocm)" ]] || return 0
    local absent
    absent="$(host_rocm_optional_absent)"
    [[ -n "$absent" ]] || return 0

    local -a pkgs=()
    local item
    for item in $absent; do
        # Unquoted on purpose: host_apt_rocm_pkg echoes nothing when the
        # component is already installed or unavailable, and an empty
        # element would become a bogus "" argument to the installer.
        # shellcheck disable=SC2207
        case "${item}:${DISTRO_FAMILY}" in
            rocWMMA:fedora) pkgs+=("rocwmma-devel") ;;
            rocWMMA:debian) pkgs+=($(host_apt_rocm_pkg $(host_rocm_optional_names rocwmma))) ;;
            RCCL:fedora)    pkgs+=("rccl-devel") ;;
            RCCL:debian)    pkgs+=($(host_apt_rocm_pkg $(host_rocm_optional_names rccl))) ;;
        esac
    done

    log "Optional ROCm extras not installed: ${absent}"
    log "These back opt-in build toggles (rocWMMA FlashAttention, RCCL collectives)."
    if [[ ${#pkgs[@]} -eq 0 ]]; then
        # Unknown distro, or no package name resolves — say what's absent
        # and leave it there rather than guessing at a command.
        return 0
    fi

    local -a inst_cmd
    case "$DISTRO_FAMILY" in
        fedora) inst_cmd=(dnf install -y) ;;
        debian) inst_cmd=(apt-get install -y) ;;
        *)      return 0 ;;
    esac
    log "Run: sudo ${inst_cmd[0]} ${inst_cmd[1]} ${pkgs[*]}"
    if prompt_confirm "Install them now? (optional — the build works without them)"; then
        if run_sudo "${inst_cmd[@]}" "${pkgs[@]}"; then
            ok "Optional ROCm extras installed"
        else
            warn "Optional ROCm extras failed to install — the toggles that need them will stay unavailable."
        fi
    fi
    return 0
}

# Map a GPU backend to the package list needed for llama.cpp to compile
# against it on this distro family. Echoes a space-separated list (empty
# if everything is already present, or if we don't know how to detect
# packages on this distro). Returns 0 always.
host_missing_gpu_sdk_packages() {
    local backend="$1"
    local -a need=()

    case "$backend" in
        rocm)
            # A usable toolchain on disk beats any package-name check —
            # ROCm gets installed from AMD's repo, from the distro, or
            # from source, and only the first of those matches the names
            # below. See host_rocm_sdk_usable.
            if host_rocm_sdk_usable; then
                echo ""
                return 0
            fi
            case "$DISTRO_FAMILY" in
                fedora)
                    # Fedora's native ROCm packages. Versioned together by
                    # the distro, so dnf resolves a consistent set.
                    for pkg in rocm-hip-devel rocblas-devel hipblas-devel rocm-cmake; do
                        rpm -q "$pkg" >/dev/null 2>&1 || need+=("$pkg")
                    done
                    ;;
                debian)
                    # Two naming schemes are in play: AMD's own repo
                    # (rocblas-dev) and Debian/Ubuntu's native packaging in
                    # universe (librocblas-dev). Which one applies depends
                    # on which repos the host has, so ask apt per component
                    # instead of hardcoding either — hardcoding AMD's names
                    # is what made `apt-get install` fail with "Unable to
                    # locate package" on distro-packaged hosts.
                    local candidates pkg_choice
                    local -a unresolved=() components=()
                    #
                    # hipcc-rocm is listed as an alternative to hipcc, not
                    # a preference: both ship /usr/bin/hipcc and conflict,
                    # so on a host that already has one, naming the other
                    # makes apt swap a working compiler for an identical
                    # one. Installing fresh still picks hipcc — that is the
                    # package that pulls the matching clang and device libs.
                    if host_rocm_prefer_amd_packages; then
                        components=(
                            "hipcc hipcc-rocm"
                            "rocm-hip-runtime-dev libamdhip64-dev"
                            "rocblas-dev librocblas-dev"
                            "hipblas-dev libhipblas-dev"
                            "rocm-cmake"
                        )
                    else
                        components=(
                            "hipcc hipcc-rocm"
                            "libamdhip64-dev rocm-hip-runtime-dev"
                            "librocblas-dev rocblas-dev"
                            "libhipblas-dev hipblas-dev"
                            "rocm-cmake"
                        )
                    fi
                    for candidates in "${components[@]}"; do
                        # shellcheck disable=SC2086
                        if pkg_choice="$(host_apt_rocm_pkg $candidates)"; then
                            if [[ -n "$pkg_choice" ]]; then
                                need+=("$pkg_choice")
                            fi
                        else
                            # No repo on this host carries it under any
                            # name. Report the preferred spelling — for an
                            # existing /opt/rocm that's AMD's, and AMD's
                            # repo is what the caller offers to add next.
                            unresolved+=("${candidates%% *}")
                        fi
                    done
                    if [[ ${#unresolved[@]} -gt 0 ]]; then
                        need+=("${unresolved[@]}")
                    fi
                    ;;
            esac
            ;;
        cuda)
            # Detect by binary presence — CUDA toolkit ships nvcc. NVIDIA's
            # RPM/DEB drops it under /usr/local/cuda/bin (not on PATH);
            # Fedora's RPMFusion cuda-devel and Arch's cuda put it on PATH.
            # Treat any of these as "installed" so we don't try to dnf-install
            # a package that's already there.
            host_find_nvcc >/dev/null || need+=("cuda-toolkit")
            # nvcc rejects host compilers newer than its supported ceiling
            # (e.g. CUDA 12.x refuses gcc 16). Pull in a side-by-side gcc
            # the builder can hand to cmake via CMAKE_CUDA_HOST_COMPILER.
            # Names differ per distro — and on Fedora, the version of the
            # compat package shifts each release (Fedora 44 ships gcc15
            # only; older releases shipped gcc13/gcc14). Pick the first one
            # the distro actually offers.
            case "$DISTRO_FAMILY" in
                fedora)
                    if ! host_find_cuda_host_compiler >/dev/null 2>&1; then
                        for pkg in gcc15-c++ gcc14-c++ gcc13-c++; do
                            if dnf info "$pkg" >/dev/null 2>&1; then
                                need+=("$pkg")
                                break
                            fi
                        done
                    fi
                    # libnccl-devel enables llama.cpp's GGML_CUDA_NCCL path
                    # (optimized AllReduce for -sm tensor across multiple
                    # GPUs). The cmake option defaults ON; without the
                    # headers the build silently falls back to the slower
                    # shfl_tensor_async path. NVIDIA ships this in their
                    # cuda-fedoraN repo as libnccl-devel.
                    rpm -q libnccl-devel >/dev/null 2>&1 || need+=("libnccl-devel")
                    ;;
                debian)
                    if ! host_find_cuda_host_compiler >/dev/null 2>&1; then
                        for pkg in g++-13 g++-12 g++-14; do
                            if apt-cache show "$pkg" >/dev/null 2>&1; then
                                need+=("$pkg")
                                break
                            fi
                        done
                    fi
                    # See fedora branch above — libnccl-dev is what
                    # llama.cpp's find_package(NCCL) needs at build time.
                    dpkg -s libnccl-dev >/dev/null 2>&1 || need+=("libnccl-dev")
                    ;;
            esac
            ;;
        vulkan)
            # llama.cpp's Vulkan backend pulls in the Vulkan loader/headers
            # AND the SPIR-V C++ headers (spirv/unified1/spirv.hpp). The GPU
            # driver typically lays down the runtime loader, but the dev
            # headers and shader compiler need to be requested explicitly.
            # vulkan-tools provides vulkaninfo, which the post-install
            # backend probe (internal/builder/detect.go) uses to enumerate
            # GPUs — without it the UI marks the vulkan backend unavailable
            # even after a clean SDK install.
            case "$DISTRO_FAMILY" in
                fedora)
                    for pkg in glslc vulkan-headers vulkan-loader-devel spirv-headers-devel vulkan-tools; do
                        rpm -q "$pkg" >/dev/null 2>&1 || need+=("$pkg")
                    done
                    ;;
                debian)
                    # glslc is its own package on Debian (frontend to
                    # shaderc); glslang-tools ships glslangValidator
                    # which find_package(Vulkan) does not use.
                    for pkg in glslc libvulkan-dev spirv-headers vulkan-tools; do
                        dpkg -s "$pkg" >/dev/null 2>&1 || need+=("$pkg")
                    done
                    ;;
                *)
                    # Best-effort fallback for unknown distros: at least flag
                    # the shader compiler if it's missing.
                    command -v glslc >/dev/null 2>&1 || need+=("glslc")
                    ;;
            esac
            ;;
        cpu|metal)
            : # nothing extra
            ;;
    esac

    echo "${need[*]}"
}

# Ensure NVIDIA's CUDA dnf repository is configured. The base Fedora repos
# don't carry cuda-toolkit, so without this `dnf install cuda-toolkit` only
# works if the user manually downloaded an RPM (which is how stale 12.6
# installs end up locked to a release that doesn't support newer GPUs).
# NVIDIA publishes a per-Fedora-version repo file; we probe the running
# version first, then walk back through known-good releases. No-op if a
# cuda-fedora*.repo is already present.
host_install_nvidia_cuda_repo_fedora() {
    if compgen -G "/etc/yum.repos.d/cuda-fedora*.repo" >/dev/null 2>&1; then
        return 0
    fi

    local current_ver
    current_ver="$(rpm -E %fedora 2>/dev/null || echo 41)"

    log "NVIDIA CUDA dnf repository is not configured."
    if ! prompt_confirm "Add it now? (needed to install or upgrade cuda-toolkit on Fedora)"; then
        warn "Skipped — cuda-toolkit install will likely fail without this repo."
        return 1
    fi

    local versions=("$current_ver")
    # Walk back through Fedora releases NVIDIA has historically published.
    # Newer releases are tried first; the first reachable one wins.
    for v in 44 43 42 41; do
        [[ "$v" == "$current_ver" ]] && continue
        versions+=("$v")
    done

    local repo_url
    for ver in "${versions[@]}"; do
        repo_url="https://developer.download.nvidia.com/compute/cuda/repos/fedora${ver}/x86_64/cuda-fedora${ver}.repo"
        log "Probing $repo_url"
        if curl -fsI "$repo_url" >/dev/null 2>&1; then
            curl -fsSL "$repo_url" | run_sudo tee "/etc/yum.repos.d/cuda-fedora${ver}.repo" >/dev/null || return 1
            ok "NVIDIA CUDA repository configured (fedora${ver})"
            return 0
        fi
    done
    warn "Couldn't reach NVIDIA's CUDA repo for any tested Fedora version."
    log "Configure it manually from https://developer.nvidia.com/cuda-downloads and re-run."
    return 1
}

# Parse the installed cuda-toolkit version. Echoes the major.minor (e.g.
# "12.6") if a versioned cuda-toolkit package is installed; empty if
# nothing's installed or if rpm isn't usable. Used to nudge users off
# stale CUDA versions that don't support modern GPUs.
host_installed_cuda_version() {
    [[ "$DISTRO_FAMILY" == "fedora" ]] || return 0
    rpm -qa 'cuda-toolkit-*-config-common' 2>/dev/null \
        | sed -n 's/^cuda-toolkit-\([0-9]\+\)-\([0-9]\+\)-config-common.*/\1.\2/p' \
        | sort -V | tail -n 1
}

# Detect the highest GPU compute capability via nvidia-smi. Echoes "X.Y"
# (e.g. "12.0" for Blackwell / RTX 50xx) or empty if nvidia-smi isn't
# available. Best-effort — we use it only to scale up warnings, never to
# block an install.
host_gpu_compute_cap() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | sort -V | tail -n 1 | tr -d ' '
}

# Compare two dotted versions. Returns 0 (true) if $1 >= $2.
host_version_ge() {
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | tail -n 1)" == "$1" ]]
}

# Offer to install missing GPU SDK packages for the chosen backend.
# Returns 0 if everything is in place (or user declined), 1 on install
# failure. Distro-aware: fedora and debian get auto-install; others get
# instructions and a continue/abort prompt.
host_install_gpu_sdk() {
    local backend="$1"
    local missing
    missing="$(host_missing_gpu_sdk_packages "$backend")"

    if [[ -z "$missing" ]]; then
        case "$backend" in
            cpu|metal) ;;
            *) ok "GPU SDK ($backend): all required packages installed" ;;
        esac
        host_warn_cuda_offpath "$backend"
        host_warn_cuda_version_for_gpu "$backend"
        host_offer_rocm_optional "$backend"
        return 0
    fi

    warn "Missing $backend SDK packages on host: $missing"
    log "These are required for llama.cpp to compile against the $backend backend from the UI."

    case "$DISTRO_FAMILY" in
        fedora)
            # cuda-toolkit isn't in Fedora's base repos — make sure NVIDIA's
            # is wired up before dnf goes hunting for it.
            if [[ "$backend" == "cuda" ]] && [[ "$missing" == *cuda-toolkit* ]]; then
                host_install_nvidia_cuda_repo_fedora || true
            fi
            log "Run: sudo dnf install $missing"
            if prompt_confirm "Install now?"; then
                # shellcheck disable=SC2086
                sudo dnf install -y $missing || return 1
                ok "$backend SDK packages installed"
            else
                warn "Skipped — first llama.cpp build with the $backend profile will fail until these are installed."
            fi
            ;;
        debian)
            # Offer AMD's repo before naming packages: which names exist
            # depends on which repos are configured, so adding one changes
            # the answer. Recent Debian/Ubuntu do carry ROCm natively in
            # universe, but it trails AMD's releases — on new hardware the
            # distro version can be too old to build for the GPU.
            if [[ "$backend" == "rocm" ]] && ! host_rocm_apt_repo_configured; then
                if host_install_amd_rocm_repo_debian; then
                    missing="$(host_missing_gpu_sdk_packages "$backend")"
                fi
            fi
            # apt-get install is all-or-nothing: a single name it can't
            # locate aborts the run and installs none of the others. Split
            # the list so the resolvable packages still land, and say
            # plainly what no configured repo carries.
            local pkg
            local -a apt_pkgs=() apt_unknown=()
            # shellcheck disable=SC2086  # $missing is a space-separated list
            for pkg in $missing; do
                if host_apt_pkg_available "$pkg"; then
                    apt_pkgs+=("$pkg")
                else
                    apt_unknown+=("$pkg")
                fi
            done
            if [[ ${#apt_unknown[@]} -gt 0 ]]; then
                warn "No configured apt repo carries: ${apt_unknown[*]}"
                case "$backend" in
                    cuda)
                        log "The CUDA toolkit comes from NVIDIA's apt repo, not the distro default. See:"
                        echo "    https://developer.nvidia.com/cuda-downloads?target_os=Linux"
                        ;;
                    rocm)
                        log "Refresh the index and make sure the component carrying ROCm is enabled:"
                        echo "    sudo apt-get update"
                        echo "    sudo add-apt-repository universe   # Ubuntu ships ROCm in universe"
                        log "Or add AMD's own repo, if they publish for this release:"
                        echo "    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html"
                        ;;
                    *)
                        log "Refresh the package index and try again: sudo apt-get update"
                        ;;
                esac
            fi
            if [[ ${#apt_pkgs[@]} -eq 0 ]]; then
                warn "Nothing installable left — the first llama.cpp build with the $backend profile will fail."
                return 1
            fi
            log "Run: sudo apt-get install ${apt_pkgs[*]}"
            if prompt_confirm "Install now?"; then
                sudo apt-get install -y "${apt_pkgs[@]}" || return 1
                ok "$backend SDK packages installed"
            else
                warn "Skipped — first llama.cpp build with the $backend profile will fail until these are installed."
            fi
            ;;
        *)
            warn "Auto-install of $backend SDK is not implemented for distro family '$DISTRO_FAMILY'."
            log "Install manually using your package manager, then re-run setup.sh install --host."
            return 1
            ;;
    esac
    host_warn_cuda_offpath "$backend"
    host_warn_cuda_version_for_gpu "$backend"
    host_verify_rocm_buildable "$backend"
    host_offer_rocm_optional "$backend"
    return 0
}

# Re-probe after installing, and say so when the host still can't compile
# the HIP backend. Package names resolving is not the same as a usable
# toolchain: a host can end up with hipcc and no hip-lang-config.cmake,
# and the only sign of it is cmake failing several minutes into the first
# build from the UI. Report it here instead.
host_verify_rocm_buildable() {
    [[ "$1" == "rocm" ]] || return 0
    # enable_language(HIP) needs a clang that can target amdgcn. cmake asks
    # hipconfig where it is and otherwise looks for a bare "clang++" on
    # PATH; on Debian/Ubuntu neither lands, because the HIP clang installs
    # as /usr/lib/llvm-N/bin/clang++ with no unversioned symlink. The
    # builder passes -DCMAKE_HIP_COMPILER when it can find one — say so
    # here when it can't, since nothing else will.
    if ! host_find_hip_clang >/dev/null; then
        warn "No HIP-capable clang++ found — llama.cpp's ROCm build needs one."
        case "$DISTRO_FAMILY" in
            debian) log "On Debian/Ubuntu it comes with hipcc (which pulls the matching clang-N and rocm-device-libs-N)." ;;
            fedora) log "On Fedora it ships in rocm-llvm, at /usr/lib64/rocm/llvm/bin/clang++." ;;
        esac
    fi
    local version
    if version="$(host_rocm_version)" && ! host_version_ge "$version" "$HOST_ROCM_MIN_VERSION"; then
        warn "ROCm ${version} is too old — llama.cpp's HIP backend requires ${HOST_ROCM_MIN_VERSION} or newer."
        log "The build will stop at \"At least ROCM/HIP V6.1 is required\" no matter"
        log "which dev packages are installed. Recent GPUs need newer still: RDNA4"
        log "(gfx1201) isn't recognised by compilers before ROCm 6.3."
        case "$DISTRO_FAMILY" in
            debian) log "Ubuntu 24.04 packages ROCm 5.7. Either add AMD's ROCm repo:"
                    echo "    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html"
                    log "or move to a release whose own packages are new enough (25.10+ ship 7.x)." ;;
            fedora) log "Update rocm-hip-devel, or use AMD's repo for a newer release." ;;
        esac
    fi
    local absent
    absent="$(host_rocm_missing_cmake_pkgs)"
    [[ -n "$absent" ]] || return 0
    warn "ROCm is still not buildable: no cmake config for ${absent}"
    log "llama.cpp's HIP backend resolves these via find_package(); without"
    log "them the build fails at 'does not contain the HIP runtime CMake package'."
    case "$DISTRO_FAMILY" in
        debian) log "On Debian/Ubuntu they ship in: libamdhip64-dev librocblas-dev libhipblas-dev" ;;
        fedora) log "On Fedora they ship in: rocm-hip-devel rocblas-devel hipblas-devel" ;;
    esac
    log "Searched: $(host_rocm_prefix_candidates | tr '\n' ' ')"
    return 0
}

# Loud-warn if the installed CUDA toolkit predates the GPU's compute
# capability. RTX 50xx (Blackwell, sm_120) needs CUDA 12.8+; CUDA 12.6 will
# build but silently fall back to an older arch and won't actually run on
# the GPU. We can't auto-upgrade safely (NVIDIA's repo split keeps multiple
# versions on disk), so spell out the manual fix.
host_warn_cuda_version_for_gpu() {
    [[ "$1" == "cuda" ]] || return 0
    local cuda_ver gpu_cap
    cuda_ver="$(host_installed_cuda_version)"
    gpu_cap="$(host_gpu_compute_cap)"
    [[ -z "$cuda_ver" || -z "$gpu_cap" ]] && return 0

    # Required CUDA toolkit major.minor by GPU compute capability.
    # Blackwell needs 12.8; Hopper (9.0) was already on 11.8/12.0.
    local needed=""
    if host_version_ge "$gpu_cap" "12.0"; then
        needed="12.8"
    fi
    [[ -z "$needed" ]] && return 0

    if ! host_version_ge "$cuda_ver" "$needed"; then
        warn "CUDA toolkit $cuda_ver is older than the GPU requires (compute $gpu_cap → CUDA >= $needed)."
        log "llama.cpp will build but won't actually run on this GPU."
        # Offer to wire up NVIDIA's repo so the user can upgrade in-place.
        # We don't run the upgrade ourselves — that's a multi-package shuffle
        # and the user should see what dnf is about to do.
        if [[ "$DISTRO_FAMILY" == "fedora" ]]; then
            host_install_nvidia_cuda_repo_fedora || true
            log "To upgrade, run:"
            echo "    sudo dnf install cuda-toolkit"
        fi
    fi
}

# NVIDIA's official RPM/DEB drops nvcc at /usr/local/cuda/bin without
# touching PATH, which breaks cmake's CUDA detection for anyone running
# the build outside the llama-toolchest service (which sets the path
# explicitly in the builder). Emit a heads-up so users know how to wire
# it into their shell.
host_warn_cuda_offpath() {
    [[ "$1" == "cuda" ]] || return 0
    if command -v nvcc >/dev/null 2>&1; then
        return 0
    fi
    local nvcc_path
    nvcc_path="$(host_find_nvcc 2>/dev/null)" || return 0
    log "nvcc is installed at $nvcc_path but not on PATH."
    log "llama-toolchest's builder picks it up automatically. To run cmake/nvcc by hand, add:"
    echo "    export PATH=\"$(dirname "$nvcc_path"):\$PATH\""
}

host_build_binary() {
    local out; out="$(host_binary_path)"
    local commit; commit="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    local ver="dev-$commit"
    local date; date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    log "Building llama-toolchest from source ($ver)..."
    mkdir -p "$(dirname "$out")"
    (cd "$SCRIPT_DIR" && \
        go build \
            -ldflags="-s -w -X main.version=$ver -X main.commit=$commit -X main.date=$date" \
            -o "$out" \
            ./cmd/llama-toolchest)
    ok "Installed binary: $out"
}

# Repo coordinates for fetching released packages. Adjust if the project
# moves; everything else in this file derives from these.
readonly HOST_RELEASE_REPO="tmac1973/llama-toolchest"
readonly HOST_RELEASE_API="https://api.github.com/repos/${HOST_RELEASE_REPO}/releases/latest"
readonly HOST_RELEASE_DOWNLOAD="https://github.com/${HOST_RELEASE_REPO}/releases/download"

# Map host architecture to the suffix used in goreleaser asset names.
host_pkg_arch() {
    case "$(uname -m)" in
        x86_64)  echo "amd64" ;;
        aarch64|arm64) echo "arm64" ;;
        *) return 1 ;;
    esac
}

# Map distro family to the package extension shipped for it.
host_pkg_ext() {
    case "$DISTRO_FAMILY" in
        fedora) echo "rpm" ;;
        debian) echo "deb" ;;
        *) return 1 ;;
    esac
}

# Fetch the latest release tag from GitHub. Echoes the version (without the
# leading "v"). Honors GITHUB_TOKEN for higher rate limits.
host_latest_release_version() {
    local auth_args=()
    [[ -n "${GITHUB_TOKEN:-}" ]] && auth_args=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    local json
    json="$(curl -fsSL "${auth_args[@]}" "$HOST_RELEASE_API")" || return 1
    # Cheap JSON parse: tag_name is on its own line with the value quoted.
    echo "$json" | grep -m1 '"tag_name":' | sed 's/.*"v\?\([^"]*\)".*/\1/'
}

# Install llama-toolchest from a published release. Default version is
# whatever GitHub considers latest; can be overridden via the LT_VERSION
# env var (useful for pinning a known-good release).
host_install_from_package() {
    local arch ext
    arch="$(host_pkg_arch)" || { err "Unsupported architecture: $(uname -m)"; return 1; }
    ext="$(host_pkg_ext)"   || { err "Package install isn't supported on distro family '$DISTRO_FAMILY'. Use --from-source."; return 1; }

    local version="${LT_VERSION:-}"
    if [[ -z "$version" ]]; then
        log "Looking up latest release of $HOST_RELEASE_REPO..."
        version="$(host_latest_release_version)" || { err "Failed to query GitHub releases API. Check network or set LT_VERSION=X.Y.Z to skip the lookup."; return 1; }
        [[ -z "$version" ]] && { err "Couldn't parse a version from the release JSON."; return 1; }
    fi
    log "Installing version v$version (${arch}, .${ext})"

    local asset="llama-toolchest_${version}_linux_${arch}.${ext}"
    local pkg_url="${HOST_RELEASE_DOWNLOAD}/v${version}/${asset}"
    local sums_url="${HOST_RELEASE_DOWNLOAD}/v${version}/checksums.txt"

    local tmpdir
    tmpdir="$(mktemp -d)"
    # Best-effort cleanup; even if the function returns early we don't want
    # leftover packages eating /tmp.
    trap "rm -rf '$tmpdir'" RETURN

    log "Downloading $asset..."
    if ! curl -fsSL --output "$tmpdir/$asset" "$pkg_url"; then
        err "Download failed: $pkg_url"
        return 1
    fi

    if curl -fsSL --output "$tmpdir/checksums.txt" "$sums_url" 2>/dev/null; then
        local expected actual
        expected="$(grep " ${asset}\$" "$tmpdir/checksums.txt" | awk '{print $1}')"
        actual="$(sha256sum "$tmpdir/$asset" | awk '{print $1}')"
        if [[ -z "$expected" ]]; then
            warn "$asset not found in checksums.txt; skipping verification."
        elif [[ "$expected" != "$actual" ]]; then
            err "Checksum mismatch for $asset"
            err "  expected: $expected"
            err "  actual:   $actual"
            return 1
        else
            ok "Checksum verified"
        fi
    else
        warn "Couldn't fetch checksums.txt; skipping verification."
    fi

    # Clean up a previous from-source binary if one exists, so PATH
    # resolution doesn't keep pointing at the old user-local copy.
    if [[ -f "$HOME/.local/bin/llama-toolchest" ]]; then
        log "Removing previous from-source binary at $HOME/.local/bin/llama-toolchest"
        rm -f "$HOME/.local/bin/llama-toolchest"
    fi

    log "Installing $asset (sudo)..."
    case "$DISTRO_FAMILY" in
        fedora) sudo dnf install -y "$tmpdir/$asset" || return 1 ;;
        debian) sudo apt-get install -y "$tmpdir/$asset" || return 1 ;;
    esac

    ok "Installed: $(/usr/bin/llama-toolchest --version 2>/dev/null || echo "v$version")"
}

# Echo the management UI port, preferring the listen_addr from an existing
# config file (so an upgrade install reports the actual port, not the
# default). Falls back to LLAMA_TOOLCHEST_PORT or 3000.
host_effective_port() {
    local cfg_path; cfg_path="$(host_config_path)"
    if [[ -f "$cfg_path" ]]; then
        # listen_addr looks like ":3001" or "0.0.0.0:3001"; strip everything
        # up to and including the last colon. Quotes optional in YAML.
        local addr port
        addr="$(grep -E '^[[:space:]]*listen_addr:' "$cfg_path" 2>/dev/null \
            | head -1 | sed -E 's/^[[:space:]]*listen_addr:[[:space:]]*"?([^"#]+)"?.*/\1/' | tr -d ' ')" || true
        port="${addr##*:}"
        if [[ "$port" =~ ^[0-9]+$ ]]; then
            echo "$port"
            return 0
        fi
    fi
    echo "${LLAMA_TOOLCHEST_PORT:-3000}"
}

# Write the example config to the user's config dir if it doesn't exist.
# Existing configs are left alone — we don't overwrite the user's settings.
host_write_config() {
    local cfg_path; cfg_path="$(host_config_path)"
    local cfg_dir; cfg_dir="$(host_config_dir)"
    local data_dir; data_dir="$(host_data_dir)"

    mkdir -p "$cfg_dir" "$data_dir/builds" "$data_dir/models" "$data_dir/config"

    if [[ -f "$cfg_path" ]]; then
        log "Config already exists at $cfg_path — leaving it alone."
        return 0
    fi

    log "Writing config: $cfg_path"
    cat > "$cfg_path" <<EOF
# llama-toolchest configuration (host install, $(host_scope) scope)
# Generated by setup.sh — edit this file or use the Settings UI.

listen_addr: ":${LLAMA_TOOLCHEST_PORT:-3000}"
data_dir: "$data_dir"
llama_port: ${LLAMA_TOOLCHEST_INFERENCE_PORT:-8080}
external_url: "http://localhost:${LLAMA_TOOLCHEST_PORT:-3000}"
log_level: "info"
models_max: 1
auto_start: false
EOF
}

# Write a systemd drop-in override carrying GPU env vars (e.g.
# HSA_OVERRIDE_GFX_VERSION) and, when needed, an ExecStart pointing at a
# non-standard binary path. The packaged unit file stays generic;
# per-machine knobs live in the override.
#
# For from-package installs we only write the override if there's
# something non-default to record (e.g., HSA_OVERRIDE_GFX_VERSION). The
# package's unit already invokes /usr/bin/llama-toolchest with the right
# defaults, so an empty override is just noise.
host_write_unit_override() {
    local scope; scope="$(service_scope)"
    local override_dir
    if [[ "$scope" == "system" ]]; then
        override_dir="/etc/systemd/system/llama-toolchest.service.d"
    else
        override_dir="${HOME}/.config/systemd/user/llama-toolchest.service.d"
    fi

    local need_execstart=false
    [[ "${HOST_INSTALL_MODE:-package}" == "source" ]] && need_execstart=true

    local need_env=false
    [[ -n "${AMD_GFX_VERSION:-}" ]] && need_env=true

    if [[ "$need_execstart" == false ]] && [[ "$need_env" == false ]]; then
        # Clean up any stale override left from a previous from-source install.
        local stale="$override_dir/override.conf"
        if [[ -f "$stale" ]]; then
            rm -f "$stale"
            log "Removed stale unit override"
            _systemctl daemon-reload
        fi
        return 0
    fi

    mkdir -p "$override_dir"
    local override_file="$override_dir/override.conf"

    {
        echo "[Service]"
        if [[ "$need_execstart" == true ]]; then
            echo "ExecStart="
            echo "ExecStart=$(host_binary_path) --config $(host_config_path)"
        fi
        if [[ "$need_env" == true ]]; then
            echo "Environment=HSA_OVERRIDE_GFX_VERSION=${AMD_GFX_VERSION}"
        fi
    } > "$override_file"

    log "Wrote unit override: $override_file"
    # systemd needs a reload to pick up override.conf changes; without this,
    # the next service_restart re-runs the previously cached ExecStart and
    # ignores our new binary path.
    _systemctl daemon-reload
}

host_install() {
    local mode="${HOST_INSTALL_MODE:-package}"
    log "Host install — scope: $(host_scope), mode: $mode"
    log "GPU backend: ${GPU_VENDOR:-unknown} (${GPU_INFO:-no description})"
    # HOST_SDK_BACKENDS may carry multiple entries (e.g. rocm + vulkan); fall
    # back to GPU_VENDOR for callers that don't populate it.
    local -a sdk_backends=("${HOST_SDK_BACKENDS[@]:-}")
    if [[ ${#sdk_backends[@]} -eq 0 || -z "${sdk_backends[0]}" ]]; then
        sdk_backends=()
        [[ -n "${GPU_VENDOR:-}" ]] && sdk_backends=("$GPU_VENDOR")
    fi
    if [[ ${#sdk_backends[@]} -gt 1 ]]; then
        log "Host SDKs to install: ${sdk_backends[*]}"
    fi
    echo ""

    # GPU SDK packages so llama.cpp builds from the UI succeed first try.
    # Same step for both install modes; install each backend independently
    # so a failure in one (e.g. vulkan headers missing from a stale repo)
    # doesn't block the others.
    for backend in "${sdk_backends[@]}"; do
        host_install_gpu_sdk "$backend" || \
            warn "GPU SDK ($backend) install reported issues; continuing."
    done

    case "$mode" in
        package)
            host_install_from_package || return 1
            ;;
        source)
            if ! host_check_build_toolchain; then
                err "Resolve missing build prerequisites and re-run."
                return 1
            fi
            host_build_binary
            # Verify the bin dir is on PATH (user install only).
            if [[ "$(host_scope)" == "user" ]] && ! echo ":$PATH:" | grep -q ":$(host_bin_dir):"; then
                warn "$(host_bin_dir) is not on your PATH."
                log "Add it to your shell rc (e.g. ~/.bashrc or ~/.zshrc):"
                echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
                echo ""
            fi
            ;;
        *)
            err "Unknown HOST_INSTALL_MODE: $mode (expected 'package' or 'source')"
            return 1
            ;;
    esac

    # Config skeleton — only matters for user-scope; system installs that
    # want a /etc config can copy from the .yaml.example shipped in the package.
    if [[ "$(host_scope)" == "user" ]]; then
        host_write_config
    fi

    # Install the systemd unit. For from-package installs the package
    # already shipped a unit at /usr/lib/systemd/{system,user}/, so we
    # skip the copy and just rely on the packaged one. For from-source
    # we copy our local copy to the user's config dir.
    if [[ "$mode" == "source" ]]; then
        local unit_src="$SCRIPT_DIR/packaging/systemd/llama-toolchest.service"
        if [[ "$(host_scope)" == "user" ]]; then
            unit_src="$SCRIPT_DIR/packaging/systemd/llama-toolchest.user.service"
        fi
        service_install "$unit_src"
    else
        # Package's postinstall already ran daemon-reload, but a re-install
        # with new unit content benefits from another reload to be sure.
        if [[ "$(service_scope)" == "user" ]]; then
            systemctl --user daemon-reload >/dev/null 2>&1 || true
        else
            systemctl daemon-reload >/dev/null 2>&1 || true
        fi
    fi

    host_write_unit_override

    # MIGRATE_SKIP_START: when set, the migrate command takes ownership of
    # service start/stop sequencing (it needs to write the translated
    # config + restored registry before first start). Don't prompt and
    # don't restart here — the caller will handle it.
    if [[ "${MIGRATE_SKIP_START:-0}" == "1" ]]; then
        log "Skipping service start (caller will handle)."
    elif service_is_active; then
        log "Service is running; restarting to pick up the new binary..."
        service_restart
        ok "llama-toolchest service restarted"
    elif prompt_confirm "Enable and start the service now?"; then
        service_enable
        ok "llama-toolchest service enabled and started"
    else
        log "Service installed but not enabled. Start later with:"
        if [[ "$(host_scope)" == "user" ]]; then
            echo "    systemctl --user start llama-toolchest"
        else
            echo "    sudo systemctl start llama-toolchest"
        fi
    fi

    echo ""
    ok "Host install complete."
    echo ""
    case "$mode" in
        package) echo "  Binary:      /usr/bin/llama-toolchest (system, from package)" ;;
        source)  echo "  Binary:      $(host_binary_path)" ;;
    esac
    echo "  Config:      $(host_config_path)"
    echo "  Data dir:    $(host_data_dir)"
    echo "  Web UI:      http://localhost:$(host_effective_port)"
    echo ""
}

host_uninstall() {
    log "Uninstalling host install — scope: $(host_scope)"

    service_uninstall

    local override_dir
    if [[ "$(service_scope)" == "system" ]]; then
        override_dir="/etc/systemd/system/llama-toolchest.service.d"
    else
        override_dir="${HOME}/.config/systemd/user/llama-toolchest.service.d"
    fi
    [[ -d "$override_dir" ]] && rm -rf "$override_dir"

    # Remove the binary. Two cases:
    #  (a) Installed via package — uninstall via dnf/apt so the system
    #      unit, /usr/bin binary, and the example config all go cleanly.
    #  (b) Installed from source — just rm the user-local binary.
    if rpm -q llama-toolchest >/dev/null 2>&1; then
        log "Removing llama-toolchest package (dnf, sudo)..."
        sudo dnf remove -y llama-toolchest || warn "Package removal returned non-zero; continuing."
    elif dpkg -s llama-toolchest >/dev/null 2>&1; then
        log "Removing llama-toolchest package (apt-get, sudo)..."
        sudo apt-get remove -y llama-toolchest || warn "Package removal returned non-zero; continuing."
    fi
    # Always check the user-local path too — we might have both if the user
    # switched modes without uninstalling first.
    local bin; bin="$(host_binary_path)"
    if [[ -f "$bin" ]]; then
        rm -f "$bin"
        log "Removed: $bin"
    fi

    local data_dir; data_dir="$(host_data_dir)"
    local cfg_dir; cfg_dir="$(host_config_dir)"
    log "Config dir ($cfg_dir) and data dir ($data_dir) preserved (contains your models and builds)."
    if prompt_confirm "Also remove config and data dirs? This DELETES all downloaded models and llama.cpp builds."; then
        rm -rf "$cfg_dir" "$data_dir"
        log "Removed config and data dirs."
    fi

    ok "Host uninstall complete."
}

# Returns 0 if a host install is present (binary on disk, either the
# packaged /usr/bin path or a from-source path under host_bin_dir).
# Used by setup.sh's auto-routing for up/down/logs to decide whether
# to call into the host or container path.
host_is_installed() {
    [[ -x /usr/bin/llama-toolchest ]] || [[ -x "$(host_binary_path)" ]]
}

host_up() {
    if ! host_is_installed; then
        err "No host install found. Run './setup.sh install --host' first."
        return 1
    fi
    if service_is_active; then
        ok "llama-toolchest is already running"
        return 0
    fi
    log "Starting llama-toolchest service..."
    service_start
    ok "llama-toolchest started"
}

host_down() {
    if ! service_is_active; then
        ok "llama-toolchest is already stopped"
        return 0
    fi
    log "Stopping llama-toolchest service..."
    service_stop
    ok "llama-toolchest stopped"
}

host_logs() {
    if [[ "$(service_scope)" == "user" ]]; then
        journalctl --user -u "$SERVICE_NAME" -f
    else
        journalctl -u "$SERVICE_NAME" -f
    fi
}

# Whether a backend is applicable to this host. Used by deps/status to
# avoid reporting "missing cuda-toolkit" on a machine that has no NVIDIA
# GPU. Vulkan is always applicable — it's the cross-vendor fallback.
host_backend_applicable() {
    case "$1" in
        cuda)   command -v nvidia-smi >/dev/null 2>&1 || [[ -e /dev/nvidia0 ]] ;;
        rocm)   [[ -e /dev/kfd ]] ;;
        vulkan) return 0 ;;
        *)      return 1 ;;
    esac
}

# Per-backend SDK report. Echoes one line per backend with state +
# remediation command. Skips backends that aren't applicable on this
# host (e.g. cuda when there's no NVIDIA GPU). Returns non-zero if any
# applicable backend has missing packages.
host_report_sdk_deps() {
    local exit_code=0
    local backend missing inst_cmd
    case "$DISTRO_FAMILY" in
        debian) inst_cmd="sudo apt-get install -y" ;;
        fedora) inst_cmd="sudo dnf install -y" ;;
        *)      inst_cmd="<distro $DISTRO_FAMILY: install manually>" ;;
    esac
    for backend in cuda rocm vulkan; do
        if ! host_backend_applicable "$backend"; then
            printf "    %-7s %s\n" "$backend" "n/a (no matching GPU detected)"
            continue
        fi
        missing="$(host_missing_gpu_sdk_packages "$backend")"
        if [[ -z "$missing" ]]; then
            local optional=""
            if [[ "$backend" == "rocm" ]]; then
                optional="$(host_rocm_optional_absent)"
            fi
            if [[ -n "$optional" ]]; then
                printf "    %-7s %s\n" "$backend" "OK (optional not installed: $optional)"
            else
                printf "    %-7s %s\n" "$backend" "OK"
            fi
        else
            printf "    %-7s missing: %s\n" "$backend" "$missing"
            printf "            %s %s\n" "$inst_cmd" "$missing"
            exit_code=1
        fi
    done
    return $exit_code
}

# Build/runtime toolchain verification. cmake/ninja/git are pulled in as
# package depends by the .deb/.rpm so they should always be present after
# a from-package install; we still re-check because users on from-source
# may have skipped the install dance.
host_report_toolchain_deps() {
    local exit_code=0
    local item bin pkgs_debian pkgs_fedora
    # name | binary | debian package | fedora package
    local -a checks=(
        "cmake|cmake|cmake|cmake"
        "ninja|ninja|ninja-build|ninja-build"
        "git|git|git|git"
        "gcc|cc|build-essential|gcc-c++ make"
    )
    local inst_cmd
    case "$DISTRO_FAMILY" in
        debian) inst_cmd="sudo apt-get install -y" ;;
        fedora) inst_cmd="sudo dnf install -y" ;;
        *)      inst_cmd="<distro $DISTRO_FAMILY>" ;;
    esac
    for c in "${checks[@]}"; do
        IFS='|' read -r name bin pkg_d pkg_f <<<"$c"
        if command -v "$bin" >/dev/null 2>&1; then
            printf "    %-7s %s\n" "$name" "OK"
        else
            local pkg="$pkg_d"
            [[ "$DISTRO_FAMILY" == "fedora" ]] && pkg="$pkg_f"
            printf "    %-7s missing\n" "$name"
            printf "            %s %s\n" "$inst_cmd" "$pkg"
            exit_code=1
        fi
    done
    return $exit_code
}

# Top-level `deps --host` entry point. Returns 0 if everything's healthy,
# 1 if anything's missing.
host_deps() {
    local rc=0
    echo "Host install dependencies (scope: $(host_scope), distro: ${DISTRO_FAMILY:-unknown}):"
    echo ""
    echo "  Package:"
    if dpkg -s llama-toolchest >/dev/null 2>&1; then
        printf "    %-20s %s\n" "llama-toolchest" "OK ($(dpkg-query -W -f='${Version}' llama-toolchest))"
    elif rpm -q llama-toolchest >/dev/null 2>&1; then
        printf "    %-20s %s\n" "llama-toolchest" "OK ($(rpm -q --qf '%{VERSION}' llama-toolchest))"
    elif [[ -x "$(host_binary_path)" ]]; then
        printf "    %-20s %s\n" "llama-toolchest" "OK (from-source: $(host_binary_path))"
    else
        printf "    %-20s %s\n" "llama-toolchest" "not installed"
        echo "            ./setup.sh install --host"
        rc=1
    fi
    echo ""
    echo "  Build toolchain (for compiling llama.cpp from the UI):"
    host_report_toolchain_deps || rc=1
    echo ""
    echo "  Backend SDKs:"
    host_report_sdk_deps || rc=1
    echo ""
    echo "  Benchmark integration (optional — for llama-benchy presets):"
    host_report_uv_dep
    echo ""
    if [[ $rc -eq 0 ]]; then
        ok "All host dependencies satisfied."
    else
        warn "One or more dependencies are missing — see commands above."
    fi
    return $rc
}

# uv (with `uvx`) is required by the llama-benchy benchmark integration —
# the runner shells out to `uvx llama-benchy`. Missing uv only disables
# the llama-benchy preset family on the Benchmarks page; the internal
# API benchmarks still work. Reports OK or WARN, never fails the deps
# check.
host_report_uv_dep() {
    if command -v uvx >/dev/null 2>&1; then
        local ver
        ver="$(uv --version 2>/dev/null | awk '{print $2}')"
        printf "    %-7s %s\n" "uv" "OK${ver:+ ($ver)}"
    else
        printf "    %-7s %s\n" "uv" "missing — llama-benchy benchmark presets won't run"
        echo "            curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
}

host_status() {
    echo "Scope:       $(host_scope)"

    # Show whichever binary is actually present. Package install lives at
    # /usr/bin; from-source lives in the user's bin dir.
    local pkg_marker="" src_marker=""
    if rpm -q llama-toolchest >/dev/null 2>&1; then
        pkg_marker="  (package: $(rpm -q --qf '%{VERSION}' llama-toolchest))"
    elif dpkg -s llama-toolchest >/dev/null 2>&1; then
        pkg_marker="  (package: $(dpkg-query -W -f='${Version}' llama-toolchest))"
    fi
    if [[ -x /usr/bin/llama-toolchest ]]; then
        echo "Binary:      /usr/bin/llama-toolchest${pkg_marker}"
    fi
    if [[ -x "$(host_binary_path)" ]]; then
        src_marker="  (from-source)"
        echo "Binary:      $(host_binary_path)${src_marker}"
    fi
    if [[ ! -x /usr/bin/llama-toolchest ]] && [[ ! -x "$(host_binary_path)" ]]; then
        echo "Binary:      (not installed)"
    fi

    echo "Config:      $(host_config_path)$( [[ -f "$(host_config_path)" ]] && echo "" || echo "  (missing)")"
    echo "Data dir:    $(host_data_dir)"
    echo "Service:     $(service_unit_path)"
    echo ""
    echo "Backend SDKs:"
    host_report_sdk_deps || true
    echo ""
    service_status
}
