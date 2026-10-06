#!/usr/bin/env bash
# scripts/test-lib.sh
#
# Plain-bash tests for the installer libraries (scripts/lib/*.sh). They
# cover the functions that compute something from their inputs and
# environment: scope and path helpers, version and port parsing, package
# name choice, ROCm prefix probing and the migration config rewrite.
# Commands that would touch the system (systemctl, nvidia-smi, rpm,
# apt-cache, uname, ...) are replaced with shell functions, and every file
# the tests write goes into a temporary directory.
#
# Run:   make installer-test     (or: bash scripts/test-lib.sh)
# Exits non-zero if any check fails; prints "ALL PASS" otherwise.
# No dependencies beyond bash and coreutils; the migration path tests also
# need jq and are skipped without it.

set -u

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$TEST_DIR")"
LIB_DIR="${TEST_DIR}/lib"

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

# Results go to a file so checks made inside subshells still count.
RESULTS="${TMP_ROOT}/results"
: > "$RESULTS"

# The libraries print through these helpers from setup.sh. Keep them
# quiet so the test output only shows check results.
log()  { :; }
ok()   { :; }
warn() { :; }
err()  { :; }

pass() { echo "PASS" >> "$RESULTS"; }
fail() {
    echo "FAIL" >> "$RESULTS"
    echo "FAIL: $*" >&2
}

# check_eq <name> <expected> <actual>
check_eq() {
    if [[ "$2" == "$3" ]]; then
        pass
    else
        fail "$1"$'\n'"    expected: [$2]"$'\n'"    actual:   [$3]"
    fi
}

# check_true <name> <command...> — passes when the command returns 0.
check_true() {
    local name="$1"; shift
    if "$@"; then pass; else fail "$name (returned non-zero)"; fi
}

# check_false <name> <command...> — passes when the command returns non-zero.
check_false() {
    local name="$1"; shift
    if "$@"; then fail "$name (returned 0)"; else pass; fi
}

skip() { echo "SKIP: $*"; }

# A fresh HOME per run, so path helpers never point at the real one.
export HOME="${TMP_ROOT}/home"
mkdir -p "$HOME"
unset XDG_CONFIG_HOME XDG_DATA_HOME ROCM_PATH LLAMA_TOOLCHEST_PORT \
      LLAMA_TOOLCHEST_INFERENCE_PORT AMD_GFX_VERSION CONTAINER_CMD \
      HOST_INSTALL_MODE DISTRO_FAMILY

# shellcheck source=scripts/lib/service.sh
source "${LIB_DIR}/service.sh"
# shellcheck source=scripts/lib/host.sh
source "${LIB_DIR}/host.sh"
# shellcheck source=scripts/lib/migrate.sh
source "${LIB_DIR}/migrate.sh"

# quadlet_dir lives in setup.sh, which runs main() when sourced. Load only
# that function, plus the two directory constants it reads.
QUADLET_USER_DIR="${HOME}/.config/containers/systemd"
QUADLET_SYSTEM_DIR="/etc/containers/systemd"
eval "$(sed -n '/^quadlet_dir() {/,/^}/p' "${REPO_DIR}/setup.sh")"

if [[ $EUID -eq 0 ]]; then
    MY_SCOPE="system"
else
    MY_SCOPE="user"
fi

# ─── Scope ───────────────────────────────────────────────────────────────────

test_scope_current_user() {
    check_eq "service_scope matches EUID" "$MY_SCOPE" "$(service_scope)"
    if [[ "$MY_SCOPE" == "user" ]]; then
        check_eq "service_unit_path (user)" \
            "${HOME}/.config/systemd/user/llama-toolchest.service" "$(service_unit_path)"
        check_eq "quadlet_dir (user)" "$QUADLET_USER_DIR" "$(quadlet_dir)"
        check_eq "host_bin_dir (user)" "${HOME}/.local/bin" "$(host_bin_dir)"
        check_eq "host_config_dir (user, no XDG)" \
            "${HOME}/.config/llama-toolchest" "$(host_config_dir)"
        check_eq "host_data_dir (user, no XDG)" \
            "${HOME}/.local/share/llama-toolchest" "$(host_data_dir)"
        check_eq "host_binary_path (user)" \
            "${HOME}/.local/bin/llama-toolchest" "$(host_binary_path)"
        check_eq "host_config_path (user)" \
            "${HOME}/.config/llama-toolchest/llama-toolchest.yaml" "$(host_config_path)"
        check_eq "host_config_dir honours XDG_CONFIG_HOME" "/x/cfg/llama-toolchest" \
            "$(XDG_CONFIG_HOME=/x/cfg host_config_dir)"
        check_eq "host_data_dir honours XDG_DATA_HOME" "/x/data/llama-toolchest" \
            "$(XDG_DATA_HOME=/x/data host_data_dir)"

        (
            systemctl() { echo "systemctl $*"; }
            check_eq "service_systemctl adds --user" \
                "systemctl --user daemon-reload" "$(service_systemctl daemon-reload)"
            check_eq "service_systemctl passes every argument" \
                "systemctl --user enable --now llama-toolchest.service" \
                "$(service_systemctl enable --now "$SERVICE_NAME")"
        )
    fi
}

# EUID is read-only in bash, so the system scope is checked by re-running
# the helpers as root inside an unprivileged user namespace.
test_scope_system() {
    if [[ "$MY_SCOPE" == "system" ]]; then
        skip "already root; system scope covered by the current-user checks"
        return
    fi
    if ! command -v unshare >/dev/null 2>&1 || ! unshare -r true 2>/dev/null; then
        skip "unshare -r not available; system scope not tested"
        return
    fi
    local out
    out="$(HOME="$HOME" unshare -r bash -c '
        set -u
        log() { :; }; ok() { :; }; warn() { :; }; err() { :; }
        source "$1/service.sh"; source "$1/host.sh"
        QUADLET_USER_DIR=u QUADLET_SYSTEM_DIR=/etc/containers/systemd
        eval "$(sed -n "/^quadlet_dir() {/,/^}/p" "$2/setup.sh")"
        systemctl() { echo "systemctl $*"; }
        echo "$(service_scope)|$(service_unit_path)|$(quadlet_dir)|$(host_bin_dir)|$(host_config_dir)|$(host_data_dir)|$(service_systemctl daemon-reload)"
    ' _ "$LIB_DIR" "$REPO_DIR")"
    check_eq "system scope as root" \
        "system|/etc/systemd/system/llama-toolchest.service|/etc/containers/systemd|/usr/local/bin|/etc/llama-toolchest|/var/lib/llama-toolchest|systemctl daemon-reload" \
        "$out"
}

# ─── Versions ────────────────────────────────────────────────────────────────

test_version_ge() {
    check_true  "12.6 >= 12.6"   host_version_ge 12.6 12.6
    check_true  "12.8 >= 12.6"   host_version_ge 12.8 12.6
    check_true  "1.10 >= 1.9"    host_version_ge 1.10 1.9
    check_true  "13.0 >= 12.9"   host_version_ge 13.0 12.9
    check_true  "6.4.1 >= 6.4"   host_version_ge 6.4.1 6.4
    check_false "12.4 >= 12.6"   host_version_ge 12.4 12.6
    check_false "1.9 >= 1.10"    host_version_ge 1.9 1.10
    check_false "6.0 >= 6.1"     host_version_ge 6.0 6.1
}

test_installed_cuda_version() (
    rpm() {
        printf '%s\n' \
            cuda-toolkit-12-4-config-common-12.4.127-1.noarch \
            cuda-toolkit-12-10-config-common-12.10.1-1.noarch \
            cuda-toolkit-12-8-config-common-12.8.90-1.noarch \
            cuda-toolkit-config-common-12.8.90-1.noarch
    }
    DISTRO_FAMILY=debian
    check_eq "host_installed_cuda_version skips non-Fedora" "" "$(host_installed_cuda_version)"
    DISTRO_FAMILY=fedora
    check_eq "host_installed_cuda_version picks the newest" "12.10" "$(host_installed_cuda_version)"
    rpm() { :; }
    check_eq "host_installed_cuda_version with nothing installed" "" "$(host_installed_cuda_version)"
)

test_gpu_compute_cap() (
    nvidia-smi() { printf '8.6\n12.0\n7.5\n'; }
    check_eq "host_gpu_compute_cap picks the highest" "12.0" "$(host_gpu_compute_cap)"
    nvidia-smi() { printf '8.6\n8.6\n8.6\n'; }
    check_eq "host_gpu_compute_cap with identical GPUs" "8.6" "$(host_gpu_compute_cap)"
    nvidia-smi() { printf '8.9 \n'; }
    check_eq "host_gpu_compute_cap strips spaces" "8.9" "$(host_gpu_compute_cap)"
)

test_rocm_version() (
    local fake="${TMP_ROOT}/hipconfig"
    host_find_rocm_tool() { echo "$fake"; }
    printf '#!/bin/sh\necho "6.3.42131-fa1d09cbd"\n' > "$fake"; chmod +x "$fake"
    check_eq "host_rocm_version reads major.minor" "6.3" "$(host_rocm_version)"
    printf '#!/bin/sh\necho "no version here"\n' > "$fake"
    check_false "host_rocm_version rejects output without a version" host_rocm_version
    host_find_rocm_tool() { return 1; }
    check_false "host_rocm_version without hipconfig" host_rocm_version
)

# ─── Packages ────────────────────────────────────────────────────────────────

test_pkg_arch_ext() (
    uname() { echo "$FAKE_ARCH"; }
    FAKE_ARCH=x86_64;  check_eq "host_pkg_arch x86_64"  "amd64" "$(host_pkg_arch)"
    FAKE_ARCH=aarch64; check_eq "host_pkg_arch aarch64" "arm64" "$(host_pkg_arch)"
    FAKE_ARCH=arm64;   check_eq "host_pkg_arch arm64"   "arm64" "$(host_pkg_arch)"
    FAKE_ARCH=riscv64; check_false "host_pkg_arch unknown" host_pkg_arch

    DISTRO_FAMILY=fedora; check_eq "host_pkg_ext fedora" "rpm" "$(host_pkg_ext)"
    DISTRO_FAMILY=debian; check_eq "host_pkg_ext debian" "deb" "$(host_pkg_ext)"
    DISTRO_FAMILY=arch;   check_false "host_pkg_ext arch" host_pkg_ext
    DISTRO_FAMILY="";     check_false "host_pkg_ext empty" host_pkg_ext
)

test_apt_helpers() (
    apt-cache() {
        case "$2" in
            good)  printf 'good:\n  Installed: (none)\n  Candidate: 6.4.0-1\n' ;;
            none)  printf 'none:\n  Installed: (none)\n  Candidate: (none)\n' ;;
            *)     return 100 ;;
        esac
    }
    check_true  "host_apt_pkg_available with a candidate" host_apt_pkg_available good
    check_false "host_apt_pkg_available with Candidate: (none)" host_apt_pkg_available none
    check_false "host_apt_pkg_available for an unknown package" host_apt_pkg_available missing

    dpkg-query() {
        case "$3" in
            inst)    printf 'installed' ;;
            removed) printf 'config-files' ;;
            *)       return 1 ;;
        esac
    }
    check_true  "host_dpkg_installed installed" host_dpkg_installed inst
    check_false "host_dpkg_installed removed but not purged" host_dpkg_installed removed
    check_false "host_dpkg_installed unknown" host_dpkg_installed nope

    host_dpkg_installed()     { [[ " $INSTALLED " == *" $1 "* ]]; }
    host_apt_pkg_available()  { [[ " $AVAILABLE " == *" $1 "* ]]; }
    INSTALLED="" AVAILABLE="librocblas-dev"
    check_eq "host_apt_rocm_pkg picks the name apt can resolve" \
        "librocblas-dev" "$(host_apt_rocm_pkg rocblas-dev librocblas-dev)"
    INSTALLED="" AVAILABLE="rocblas-dev librocblas-dev"
    check_eq "host_apt_rocm_pkg prefers the first name" \
        "rocblas-dev" "$(host_apt_rocm_pkg rocblas-dev librocblas-dev)"
    INSTALLED="librocblas-dev" AVAILABLE="rocblas-dev"
    check_eq "host_apt_rocm_pkg says nothing when one is installed" \
        "" "$(host_apt_rocm_pkg rocblas-dev librocblas-dev)"
    check_true "host_apt_rocm_pkg returns 0 when one is installed" \
        host_apt_rocm_pkg rocblas-dev librocblas-dev
    INSTALLED="" AVAILABLE=""
    check_false "host_apt_rocm_pkg with no resolvable name" \
        host_apt_rocm_pkg rocblas-dev librocblas-dev

    host_rocm_prefer_amd_packages() { return 0; }
    check_eq "host_rocm_optional_names rocwmma, AMD first" \
        "rocwmma-dev librocwmma-dev" "$(host_rocm_optional_names rocwmma)"
    host_rocm_prefer_amd_packages() { return 1; }
    check_eq "host_rocm_optional_names rccl, distro first" \
        "librccl-dev rccl-dev" "$(host_rocm_optional_names rccl)"
    check_false "host_rocm_optional_names unknown" host_rocm_optional_names foo
)

test_backend_applicable() {
    check_true  "host_backend_applicable vulkan" host_backend_applicable vulkan
    check_false "host_backend_applicable unknown" host_backend_applicable metal
}

# ─── ROCm prefix probing ─────────────────────────────────────────────────────

test_rocm_prefix_candidates() (
    host_rocm_prefix()    { return 1; }
    host_find_rocm_tool() { return 1; }
    check_eq "candidates with nothing found" \
        $'/opt/rocm\n/usr' "$(host_rocm_prefix_candidates)"

    ROCM_PATH=/custom/rocm
    host_rocm_prefix()    { echo /custom/rocm; }
    host_find_rocm_tool() {
        case "$1" in
            hipconfig) echo /usr/bin/hipconfig ;;
            hipcc)     echo /opt/rocm-6.4.0/bin/hipcc ;;
        esac
    }
    check_eq "candidates are ordered and de-duplicated" \
        $'/custom/rocm\n/usr\n/opt/rocm-6.4.0\n/opt/rocm' "$(host_rocm_prefix_candidates)"
)

test_find_rocm_tool_and_prefix() (
    local root="${TMP_ROOT}/rocm"
    local tool="llt-test-tool-$$"
    mkdir -p "$root/bin"
    printf '#!/bin/sh\n' > "$root/bin/$tool"; chmod +x "$root/bin/$tool"

    ROCM_PATH="$root"
    check_eq "host_find_rocm_tool looks in ROCM_PATH/bin" "$root/bin/$tool" "$(host_find_rocm_tool "$tool")"
    check_false "host_find_rocm_tool misses an absent tool" host_find_rocm_tool "llt-absent-$$"
    check_eq "host_rocm_prefix prefers ROCM_PATH" "$root" "$(host_rocm_prefix)"
)

test_rocm_cmake_pkgs() (
    local a="${TMP_ROOT}/pfx-a" b="${TMP_ROOT}/pfx-b"
    mkdir -p "$a/lib/cmake/hip-lang" "$a/lib64/cmake/hip" "$b/lib/x86_64-linux-gnu/cmake/rocblas"
    touch "$a/lib/cmake/hip-lang/hip-lang-config.cmake" \
          "$a/lib64/cmake/hip/hip-config.cmake" \
          "$b/lib/x86_64-linux-gnu/cmake/rocblas/rocblas-config.cmake"
    host_rocm_prefix_candidates() { printf '%s\n' "$a" "" "$b"; }
    check_true  "cmake pkg in lib/cmake"      host_rocm_have_cmake_pkg hip-lang
    check_true  "cmake pkg in lib64/cmake"    host_rocm_have_cmake_pkg hip
    check_true  "cmake pkg in multiarch dir"  host_rocm_have_cmake_pkg rocblas
    check_false "cmake pkg absent"            host_rocm_have_cmake_pkg hipblas
    check_eq "host_rocm_missing_cmake_pkgs" "hipblas" "$(host_rocm_missing_cmake_pkgs)"
)

test_rocm_optional_absent() (
    local p="${TMP_ROOT}/pfx-opt"
    mkdir -p "$p/include"
    host_rocm_prefix() { echo "$p"; }
    check_eq "both optional extras absent" "rocWMMA RCCL" "$(host_rocm_optional_absent)"
    mkdir -p "$p/include/rocwmma"; touch "$p/include/rocwmma/rocwmma.hpp" "$p/include/rccl.h"
    check_eq "both optional extras present" "" "$(host_rocm_optional_absent)"
    host_rocm_prefix() { return 1; }
    check_eq "no ROCm install" "" "$(host_rocm_optional_absent)"
)

# ─── Config files ────────────────────────────────────────────────────────────

test_effective_port() (
    local cfg; cfg="$(host_config_path)"
    rm -f "$cfg"
    check_eq "port default" "3000" "$(host_effective_port)"
    check_eq "port from env" "4100" "$(LLAMA_TOOLCHEST_PORT=4100 host_effective_port)"

    mkdir -p "$(dirname "$cfg")"
    local line expected
    while IFS='|' read -r line expected; do
        printf 'data_dir: "/x"\n%s\nlog_level: info\n' "$line" > "$cfg"
        check_eq "port from [$line]" "$expected" "$(host_effective_port)"
    done <<'EOF'
listen_addr: ":3001"|3001
listen_addr: :3002|3002
listen_addr: "0.0.0.0:3003"|3003
  listen_addr:   "[::]:3004"  # comment|3004
listen_addr: ""|3000
listen_addr: "localhost"|3000
EOF
    rm -f "$cfg"
)

test_write_config() (
    local cfg; cfg="$(host_config_path)"
    rm -f "$cfg"
    LLAMA_TOOLCHEST_PORT=3555
    host_write_config
    check_true "host_write_config creates the file" test -f "$cfg"
    check_eq "host_write_config port" "3555" "$(host_effective_port)"
    check_true "host_write_config creates the models dir" test -d "$(host_data_dir)/models"
    echo "# edited" > "$cfg"
    host_write_config
    check_eq "host_write_config leaves an existing file alone" "# edited" "$(cat "$cfg")"
    rm -f "$cfg"
)

test_write_unit_override() (
    service_systemctl() { echo "$*" >> "${TMP_ROOT}/systemctl.log"; }
    local dir file
    dir="$(service_unit_path).d"
    file="$dir/override.conf"

    HOST_INSTALL_MODE=source AMD_GFX_VERSION=11.0.0
    host_write_unit_override
    check_eq "override (source + gfx)" \
        "[Service]
ExecStart=
ExecStart=$(host_binary_path) --config $(host_config_path)
Environment=HSA_OVERRIDE_GFX_VERSION=11.0.0" "$(cat "$file" 2>/dev/null)"

    HOST_INSTALL_MODE=package AMD_GFX_VERSION=10.3.0
    host_write_unit_override
    check_eq "override (package + gfx)" \
        "[Service]
Environment=HSA_OVERRIDE_GFX_VERSION=10.3.0" "$(cat "$file" 2>/dev/null)"

    HOST_INSTALL_MODE=package; unset AMD_GFX_VERSION
    host_write_unit_override
    check_false "override removed when nothing to record" test -e "$file"
    check_eq "daemon-reload after each change" \
        $'daemon-reload\ndaemon-reload\ndaemon-reload' "$(cat "${TMP_ROOT}/systemctl.log")"
)

# ─── Migration ───────────────────────────────────────────────────────────────

test_migrate_config() (
    local src="${TMP_ROOT}/src.yaml" dst="${TMP_ROOT}/dst.yaml"
    cat > "$src" <<'EOF'
# test config
listen_addr: ":3456"
data_dir: "/home/u/.local/share/llama-toolchest"
models_dir: "/srv/models"
llama_port: 8080
external_url: "https://ai.example.test"
hf_token: "hf_abc"
api_key: "sk-123"
log_level: "debug"
models_max: 3
auto_start: false
EOF
    LLAMA_TOOLCHEST_PORT=""
    LLAMA_TOOLCHEST_INFERENCE_PORT=8181
    migrate_write_translated_config "$src" "$dst" to-container
    check_eq "to-container sets LLAMA_TOOLCHEST_PORT from listen_addr" "3456" "$LLAMA_TOOLCHEST_PORT"
    check_eq "to-container body" \
'listen_addr: ":3000"
data_dir: "/data"
models_dir: "/srv/models"
llama_port: 8181
external_url: "https://ai.example.test"
hf_token: "hf_abc"
api_key: "sk-123"
log_level: "debug"
models_max: 3
auto_start: false' "$(sed '1,2d' "$dst")"
    check_false "_migrate_get is removed afterwards" declare -F _migrate_get

    # Container to host: the host port comes from the container's mapping.
    CONTAINER_CMD=fake_runtime
    fake_runtime() {
        cat <<'JSON'
            "PortBindings": {
                "3000/tcp": [
                    {
                        "HostIp": "",
                        "HostPort": "3999"
                    }
JSON
    }
    unset LLAMA_TOOLCHEST_INFERENCE_PORT
    printf 'listen_addr: ":3000"\ndata_dir: "/data"\n' > "$src"
    migrate_write_translated_config "$src" "$dst" to-host
    check_eq "to-host body with defaults" \
"listen_addr: \":3999\"
data_dir: \"$(host_data_dir)\"
llama_port: 8080
external_url: \"\"
hf_token: \"\"
api_key: \"\"
log_level: \"info\"
models_max: 1
auto_start: true" "$(sed '1,2d' "$dst")"

    unset CONTAINER_CMD
    migrate_write_translated_config "$src" "$dst" to-host
    check_eq "to-host without a runtime falls back to :3000" \
        'listen_addr: ":3000"' "$(sed -n 3p "$dst")"

    check_false "unknown direction fails" migrate_write_translated_config "$src" "$dst" sideways
)

test_migrate_model_paths() (
    if ! command -v jq >/dev/null 2>&1; then
        skip "jq not installed; migrate_translate_model_paths not tested"
        return
    fi
    local f="${TMP_ROOT}/models.json"
    cat > "$f" <<'EOF'
{"models": [{"id": "a", "file_path": "/data/models/a.gguf"}],
 "configs": {
   "c1": {"mmproj_path": "/data/models/a-mmproj.gguf", "draft_model_path": "/data/models/sub/d.gguf"},
   "c2": {"mmproj_path": "/data/models2/b.gguf"},
   "c3": {"name": "plain"}
 }}
EOF
    migrate_translate_model_paths "$f" "/data/models/" "/home/u/models"
    check_eq "mmproj path rewritten" "/home/u/models/a-mmproj.gguf" "$(jq -r '.configs.c1.mmproj_path' "$f")"
    check_eq "draft path rewritten" "/home/u/models/sub/d.gguf" "$(jq -r '.configs.c1.draft_model_path' "$f")"
    check_eq "similar prefix left alone" "/data/models2/b.gguf" "$(jq -r '.configs.c2.mmproj_path' "$f")"
    check_eq "config without paths unchanged" '{"name":"plain"}' "$(jq -c '.configs.c3' "$f")"
    check_eq "model file_path untouched" "/data/models/a.gguf" "$(jq -r '.models[0].file_path' "$f")"

    check_true "missing file is not an error" \
        migrate_translate_model_paths "${TMP_ROOT}/absent.json" /a /b

    # jq prints its own parse error; keep it out of the test output.
    quiet_translate() { migrate_translate_model_paths "$@" 2>/dev/null; }
    printf 'not json' > "$f"
    check_false "invalid JSON reports failure" quiet_translate "$f" /a /b
    check_eq "invalid JSON left unchanged" "not json" "$(cat "$f")"
)

# ─── Run ─────────────────────────────────────────────────────────────────────

test_scope_current_user
test_scope_system
test_version_ge
test_installed_cuda_version
test_gpu_compute_cap
test_rocm_version
test_pkg_arch_ext
test_apt_helpers
test_backend_applicable
test_rocm_prefix_candidates
test_find_rocm_tool_and_prefix
test_rocm_cmake_pkgs
test_rocm_optional_absent
test_effective_port
test_write_config
test_write_unit_override
test_migrate_config
test_migrate_model_paths

passes="$(grep -c '^PASS$' "$RESULTS")"
fails="$(grep -c '^FAIL$' "$RESULTS")"
echo "${passes} passed, ${fails} failed"
if [[ "$fails" -ne 0 ]]; then
    exit 1
fi
echo "ALL PASS"
