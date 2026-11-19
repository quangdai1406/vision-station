#!/usr/bin/env bash

set -u

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <station data folder> <laptop store folder>" >&2
    exit 2
fi

LOCAL="$1"
DEST="$2"

if [ ! -d "$LOCAL" ]; then
    echo "station data folder not found: $LOCAL" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCK_FILE="$LOCAL/.evidence_sync.lock"

if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required (util-linux)" >&2
    exit 2
fi

exec 9>"$LOCK_FILE" || {
    echo "cannot open lock file: $LOCK_FILE" >&2
    exit 2
}
if ! flock -n 9; then
    echo "another evidence sync is running; skipping this run" >&2
    exit 75
fi

cd "$REPO_ROOT" || exit 2
python3 -m station.evidence_sync run --local "$LOCAL" --dest "$DEST"
exit $?
