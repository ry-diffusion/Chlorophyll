# merc-run.sh [fresh]: on the phone (sh /var/root/merc-run.sh). Restarts the Mercury poster (with a fresh shader cache when $1 = fresh: needed after
# changing a switch in /private/var/tmp/m2v-env that changes its shaders), runs 4 lock/unlock cycles and prints its GPU ms per frame
# from the fps meter. Do not put "MercuryPosterExtension" on the same ssh command line: the script kills every process that names it.
cd /private/var/tmp
for p in $(ps -ax -o pid,command | grep "[M]ercuryPosterExtension" | cut -c1-6); do kill -9 $p; done
[ "$1" = fresh ] && { rm -rf /private/var/tmp/m2v-cache/MercuryPosterExtension.old; mv /private/var/tmp/m2v-cache/MercuryPosterExtension /private/var/tmp/m2v-cache/MercuryPosterExtension.old 2>/dev/null; }
rm -f bgmtl.state.com.apple.MercuryPoster; [ -f bgmtl.off.com.apple.MercuryPoster ] && mv bgmtl.off.com.apple.MercuryPoster /var/root/bgmtl.off.com.apple.MercuryPoster.study
S=$(wc -l < m2v.log)
for i in 1 2 3 4; do /var/root/sweetwake lock >/dev/null 2>&1; /var/root/sweetwake awake 3 >/dev/null 2>&1; /var/root/sweetwake wake >/dev/null 2>&1; /var/root/sweetwake unlock >/dev/null 2>&1; /var/root/sweetwake awake 4 >/dev/null 2>&1; done
tail -n +$S m2v.log | grep -aE "MercuryPosterExtension\]: fps" | sed -E "s/.*fps: ([0-9]+) frames.*draws\/frame ([0-9]+).*commit-to-done ([0-9]+) us mean over ([0-9]+).*gpu ([0-9]+) ms \(max ([0-9]+)\).*/fps \1 draws \2 commit-to-done \3 us over \4; gpu \5 ms (max \6)/" | tail -12
tail -n +$S m2v.log | grep -aE "MercuryPosterExtension\]: (env|device|.*refused|.*Lost)" | cut -c1-160 | head -6
ls bgmtl.off.com.apple.MercuryPoster 2>/dev/null; ps -ax -o pid,etime,command | grep "[M]ercuryPosterExtension" | cut -c1-60
