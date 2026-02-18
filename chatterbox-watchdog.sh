#!/bin/bash
# Chatterbox TTS watchdog
# Detects non-zero exits (CUDA OOM), drops page caches from the host, restarts cleanly.
# Backs off exponentially on rapid restarts to avoid looping on startup/code errors.
# Runs as a systemd service (chatterbox-watchdog.service).

RESTART_COUNT=0
LAST_RESTART_TIME=0
BACKOFF_THRESHOLD=3   # Consecutive fast restarts before backing off
MIN_UPTIME=45         # Seconds — restarts faster than this are treated as crash-loops

while true; do
    # Wait until the container exists and is running
    while ! docker ps --format '{{.Names}}' | grep -q '^chatterbox-tts$'; do
        sleep 5
    done

    START_TIME=$(date +%s)
    echo "[$(date)] chatterbox-watchdog: monitoring chatterbox-tts..."

    # Block until the container exits
    EXIT_CODE=$(docker wait chatterbox-tts 2>/dev/null)

    # Exit code 0 = graceful stop (docker stop / systemctl stop). Don't restart.
    if [ "${EXIT_CODE}" = "0" ]; then
        echo "[$(date)] chatterbox-watchdog: container stopped cleanly (code 0) — not restarting"
        RESTART_COUNT=0
        continue
    fi

    # Calculate how long the container ran before crashing
    UPTIME=$(( $(date +%s) - START_TIME ))

    # Reset backoff counter if the container ran long enough (wasn't a startup crash)
    if [ "${UPTIME}" -ge "${MIN_UPTIME}" ]; then
        RESTART_COUNT=0
    fi

    RESTART_COUNT=$(( RESTART_COUNT + 1 ))

    # Exponential backoff on rapid restarts: 5s, 15s, 45s, 90s, then cap at 120s
    if [ "${RESTART_COUNT}" -gt "${BACKOFF_THRESHOLD}" ]; then
        BACKOFF=$(( (RESTART_COUNT - BACKOFF_THRESHOLD) * 30 ))
        [ "${BACKOFF}" -gt 120 ] && BACKOFF=120
        echo "[$(date)] chatterbox-watchdog: container crashed (code ${EXIT_CODE}, uptime ${UPTIME}s) — rapid restart #${RESTART_COUNT}, backing off ${BACKOFF}s..."
        sleep "${BACKOFF}"
    else
        echo "[$(date)] chatterbox-watchdog: container crashed (code ${EXIT_CODE}, uptime ${UPTIME}s) — dropping page caches and restarting..."
    fi

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
