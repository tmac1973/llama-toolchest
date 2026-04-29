#!/bin/sh
# Run before package removal (deb prerm, rpm preun).
# Stop the system service if it's running so the binary isn't held open
# during the upgrade/remove. User services and the data dir are left alone.

if command -v systemctl >/dev/null 2>&1; then
    systemctl stop llama-toolchest.service >/dev/null 2>&1 || :
fi
