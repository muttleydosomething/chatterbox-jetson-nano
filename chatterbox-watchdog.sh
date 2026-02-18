#!/bin/bash
# Chatterbox TTS watchdog
# Detects non-zero exits (CUDA OOM), drops page caches from the host, restarts cleanly.
# Runs as a systemd service (chatterbox-watchdog.service).

while true; do
    # Wait until the container exists and is running
    while ! docker ps --format '{{.Names}}' | grep -q '^chatterbox-tts$'; do
        sleep 5
    done

    echo "[$(date)] chatterbox-watchdog: monitoring chatterbox-tts..."

    # Block until the container exits
    EXIT_CODE=$(docker wait chatterbox-tts 2>/dev/null)

    # Exit code 0 = graceful stop (docker stop / systemctl stop). Don't restart.
    if [ "${EXIT_CODE}" = "0" ]; then
        echo "[$(date)] chatterbox-watchdog: container stopped cleanly (code 0) — not restarting"
        continue
    fi

    echo "[$(date)] chatterbox-watchdog: container crashed (code ${EXIT_CODE}) — dropping page caches and restarting..."

    # Drop page caches to give NvMap contiguous physical blocks for from_pretrained()
    sync
    echo 3 > /proc/sys/vm/drop_caches

    # Brief pause for memory to settle
    sleep 3

    docker start chatterbox-tts
    if [ $? -eq 0 ]; then
        echo "[$(date)] chatterbox-watchdog: restart issued — waiting for model load..."
    else
        echo "[$(date)] chatterbox-watchdog: docker start failed, will retry in 10s"
        sleep 10
    fi

    sleep 5
done
