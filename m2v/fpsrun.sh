#!/bin/bash
# SPDX-License-Identifier: MIT
# fps test cycle for the m2v trial on the phone (Poco F1 over USB SSH):
#   m2v/fpsrun.sh [--install] [--reboot] [--swipes N] [--label L]
# --install copies $GPU_METAL_WORK/out-m2v/libm2v.dylib (and out-metal/bgmtl-inject.dylib with --inject) to the phone;
# --reboot reboots (backboardd only restarts with the phone); then stale per-process kill switches are cleared, the phone
# is unlocked, N page-swipe pairs run, and the fps meter's lines (/private/var/tmp/m2v-fps) are summarised per process:
# mean frames/s over the seconds that had >= 5 frames, and the share of frame intervals <= 17 ms (60 fps) and <= 34 ms.
set -uo pipefail
K="-o UserKnownHostsFile=$HOME/.ssh/known_hosts_ios26 -o ConnectTimeout=5"
P() { timeout "${T:-20}" ssh $K root@192.168.7.2 "$@"; }
W="${GPU_METAL_WORK:-/Users/zesmoi/git/iAndroid/work-chlorophyll}"
install=0 inject=0 reboot=0 swipes=10 label=run
while [ $# -gt 0 ]; do
  case "$1" in
    --install) install=1 ;; --inject) inject=1 ;; --reboot) reboot=1 ;;
    --swipes) swipes=$2; shift ;; --label) label=$2; shift ;;
  esac; shift
done
if [ $install = 1 ]; then
  T=90 P 'cat > /usr/local/lib/bgmtl/.libm2v.tmp' < "$W/out-m2v/libm2v.dylib" &&
  P 'chmod 755 /usr/local/lib/bgmtl/.libm2v.tmp && mv /usr/local/lib/bgmtl/.libm2v.tmp /usr/local/lib/bgmtl/libm2v.dylib' || exit 1
fi
if [ $inject = 1 ]; then
  T=90 P 'cat > /usr/local/lib/bgmtl/.inject.tmp' < "$W/out-metal/bgmtl-inject.dylib" &&
  P 'chmod 755 /usr/local/lib/bgmtl/.inject.tmp && mv /usr/local/lib/bgmtl/.inject.tmp /usr/local/lib/bgmtl/bgmtl-inject.dylib' || exit 1
fi
# The RAM-boot image: boot55 + enable_skstb (lets kern.sched_update_recommended_cores be written). A normal reboot would come
# back on the installed boot partition, so reboots go through ABL fastboot and `fastboot boot` (nothing is flashed).
IMG="${M2V_BOOT_IMG:-/Users/zesmoi/git/iAndroid/ios-beryllium/shim/out/enxerto-boot-ios26full-boot55-skstb.img}"
CORES="${M2V_CORES:-0xf1}"   # CPU0 (the boot A55) + the four A75s
if [ $reboot = 1 ]; then
  P 'sync; sweetreboot fastboot' >/dev/null 2>&1
  n=0; until fastboot devices 2>/dev/null | grep -q fastboot; do n=$((n+1)); [ $n -ge 120 ] && { echo "no fastboot"; exit 1; }; python3 -c 'import time; time.sleep(2)'; done
  fastboot boot "$IMG" >/dev/null 2>&1 || { echo "fastboot boot failed"; exit 1; }
  n=0; until [ $n -ge 3 ] && T=8 P true 2>/dev/null; do n=$((n+1)); [ $n -ge 150 ] && { echo "phone did not come back"; exit 1; }; done
  for i in $(seq 1 10); do T=13 P true; done   # let SpringBoard, backboardd and the posters settle
fi
[ "$CORES" != none ] && P "/var/root/sctest kern.sched_update_recommended_cores $CORES | grep -o 'write.*'"
# stale kill switches from unclean launches (Mercury's two instances share one state file); never backboardd's
P 'cd /private/var/tmp; for b in com.apple.MercuryPoster com.apple.springboard; do if [ -f bgmtl.off.$b ] && [ -f bgmtl.m2v.$b ]; then mv bgmtl.off.$b /var/root/bgmtl.off.$b.fpsrun; rm -f bgmtl.state.$b; [ $b = com.apple.MercuryPoster ] && for p in $(ps -ax -o pid,command | grep "[M]ercuryPosterExtension" | cut -c1-6); do kill $p; done; echo "cleared $b"; fi; done; true'
P '/var/root/sweetwake wake >/dev/null 2>&1; /var/root/sweetwake unlock >/dev/null 2>&1; true'
for i in 1 2 3; do T=13 P true; done
start=$(P 'wc -l < /private/var/tmp/m2v.log' | tr -d ' ')
P '/var/root/sweetwake home >/dev/null 2>&1; true'
for i in $(seq 1 "$swipes"); do
  P '/var/root/sweetwake swipe 360 450 50 450 300 >/dev/null 2>&1; /var/root/sweetwake swipe 50 450 360 450 300 >/dev/null 2>&1; true'
done
P "tail -n +${start:-1} /private/var/tmp/m2v.log | grep 'fps:'" > "/tmp/fpsrun-$label.txt" 2>/dev/null
P "tail -n +${start:-1} /private/var/tmp/m2v.log | grep -E 'MISSING|refused|memory is gone|Lost' | cut -c1-160 | sort | uniq -c | sort -rn | head -8"
P '/var/root/sweetgpu | grep -oE "mhz=[0-9]+|busy_pct=[0-9]+|recoveries=[0-9]+|hangs=[0-9]+" | tr "\n" " "; echo'
python3 - "/tmp/fpsrun-$label.txt" "$label" <<'EOF'
import re, sys, collections
rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    m = re.search(r'\[\d+ (\S+)\]: fps: (\d+) frames/s, intervals <=17ms (\d+), <=25 (\d+), <=34 (\d+), <=100 (\d+), longer (\d+).*draws/frame (\d+)', line)
    if m: rows[m.group(1)].append(tuple(int(x) for x in m.groups()[1:]))
for proc, r in sorted(rows.items()):
    act = [x for x in r if x[0] >= 5] or r
    fps = sum(x[0] for x in act) / len(act)
    iv = [sum(x[i] for x in act) for i in range(1, 6)]
    tot = sum(iv) or 1
    print(f"{sys.argv[2]}: {proc:24s} {fps:5.1f} fps over {len(act):3d} active s, <=17ms {100*iv[0]/tot:4.0f}%, <=34ms {100*(iv[0]+iv[1]+iv[2])/tot:4.0f}%, draws/frame {sum(x[6] for x in act)//len(act)}")
EOF
