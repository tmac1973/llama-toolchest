#!/bin/sh
# Run after package install (deb postinst, rpm post).
# Reload systemd so the freshly-installed unit is visible. We deliberately do
# NOT enable or start the service here — setup.sh handles that after the
# user picks their backend and writes the config. This keeps `apt install`
# /`dnf install` purely about getting bits on disk.

if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload >/dev/null 2>&1 || :
fi
