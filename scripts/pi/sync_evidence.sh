#!/usr/bin/env bash
# Copy pending saved parts from the Pi to the laptop evidence store (plan S4).
#
# Usage:  scripts/pi/sync_evidence.sh <station data folder> <laptop store folder>
#   e.g.  scripts/pi/sync_evidence.sh /home/pi/vision-station/data /mnt/evidence/store
#
# The first argument holds store/ and the queue file. The second is the laptop folder
# that holds the L1/ tree (the trainer's store/), reached through the SMB mount below.
#
# A lock file in the station data folder makes sure two runs never overlap (for example
# a timer and a manual run). A run that finds the lock held exits with status 75 and
# does nothing. No password is in this file: the SMB credentials stay in a root-only file.
#
# SMB mount (one-time setup on the Pi; replace every <PLACEHOLDER>). The credentials file
# holds the share user and password, owned by root, mode 600, and is not in the repo:
#
#   # /etc/evidence-smb.cred      (root:root, chmod 600)
#   #   username=<SHARE_USER>
#   #   password=<SHARE_PASSWORD>
#
#   # /etc/fstab line
#   //<LAPTOP_HOST>/<SHARE_NAME>  /mnt/evidence  cifs  credentials=/etc/evidence-smb.cred,uid=<PI_USER>,gid=<PI_USER>,vers=<SMB_VERSION>,_netdev,noauto,x-systemd.automount,nofail  0  0
#
# With noauto,x-systemd.automount the share mounts on first access. If the laptop is off,
# the destination folder is missing and the run fails with "destination is not reachable";
# the parts stay queued and the next run copies them.

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
