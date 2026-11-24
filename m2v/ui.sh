#!/bin/bash
# SPDX-License-Identifier: MIT
# Drive the phone's UI over SSH and take screenshots (Poco F1, sweetwake/sweetshot):
#   m2v/ui.sh OUTDIR step...     steps: shot:NAME  wake  unlock  home  spotlight  control  tap:X:Y  swipe:X1:Y1:X2:Y2  wait:N
# SHOT_SCALE=2 takes half-size shots (default 3 = a third, which drops thin detail such as icon glyphs).
# A screenshot is retried until it is not the black frame of a sleeping display (> 3000 bytes).
set -uo pipefail
K="-o UserKnownHostsFile=$HOME/.ssh/known_hosts_ios26 -o ConnectTimeout=5"
P() { timeout 20 ssh $K root@192.168.7.2 "$@"; }
out=$1; shift
mkdir -p "$out"
for step in "$@"; do
  case "$step" in
    shot:*) f="$out/${step#shot:}.png"; n=0
            while :; do P "/var/root/sweetshot /private/var/tmp/s.png ${SHOT_SCALE:-3} >/dev/null 2>&1; cat /private/var/tmp/s.png" > "$f"
                        [ "$(stat -f%z "$f")" -gt 3000 ] && break; n=$((n+1)); [ $n -ge 6 ] && break
                        P '/var/root/sweetwake wake >/dev/null 2>&1; true'; done; echo "$f $(stat -f%z "$f")" ;;
    wake) P '/var/root/sweetwake wake >/dev/null 2>&1; true' ;;
    unlock) P '/var/root/sweetwake wake >/dev/null 2>&1; /var/root/sweetwake unlock >/dev/null 2>&1; true' ;;
    home) P '/var/root/sweetwake home >/dev/null 2>&1; true' ;;
    spotlight) P '/var/root/sweetwake swipe 207 300 207 700 300 >/dev/null 2>&1; true' ;;
    control) P '/var/root/sweetwake swipe 400 6 400 500 300 >/dev/null 2>&1; true' ;;
    tap:*) IFS=: read -r _ x y <<< "$step"; P "/var/root/sweetwake tap $x $y >/dev/null 2>&1; true" ;;
    swipe:*) IFS=: read -r _ a b c d <<< "$step"; P "/var/root/sweetwake swipe $a $b $c $d 300 >/dev/null 2>&1; true" ;;
    wait:*) for i in $(seq 1 "${step#wait:}"); do timeout 13 ssh $K root@192.168.7.2 true; done ;;
  esac
done
