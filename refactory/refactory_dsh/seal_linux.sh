# The sealed workspace's runner on Linux (`refactory_dsh.seal`), run by bash inside new
# user, network, mount and PID namespaces, where it is root of its own user namespace.
# It builds a new root from an allow-list and runs the command in it, unprivileged.
# A file, never inline code, so nothing between refactory and this shell (systemd-run
# expands $NAME in its arguments) can change what it does. Any step that fails stops
# it: nothing runs half sealed.
#
# Arguments: refactory-seal, the workspace, the caller's uid and gid, the ulimit
# options, the close step's file, the paths to bind (colon-separated), the names of
# /usr/local to keep (colon-separated, glob patterns), the command's PATH and locale;
# then whatever DSH appends, its command after "--".
[ "$1" = refactory-seal ] || { echo "refactory-seal: bad arguments" >&2; exit 97; }
ws="$2"; uid="$3"; gid="$4"; limits="$5"; closer="$6"; reads="$7"; kept="$8"
cmd_path="$9"; locale="${10}"
shift 10
fail() { echo "refactory-seal: $1" >&2; exit 97; }
exec 3<"$ws" || fail "cannot open the workspace"
# The close step's code, read now: it runs as `bash -c` (a script file would hold a
# descriptor of its own open) in the nested namespace, past anything that rewrites
# arguments.
close_step="$(cat "$closer")" && [ -n "$close_step" ] || fail "cannot read the close step"
# One more user namespace (the unprivileged one the command runs in) and none after it:
# the command cannot become root of a namespace of its own.
echo 1 > /proc/sys/user/max_user_namespaces || fail "cannot limit user namespaces"
mount -n -t tmpfs -o mode=0755,size=1m seal /tmp || fail "cannot mount /tmp"
r=/tmp/root
{ mkdir "$r" && mount -n -t tmpfs -o mode=0755,size=16m seal "$r"; } || fail "cannot make the new root"
mkdir -p "$r/etc" || fail "cannot make /etc"
# Each bound alone (--bind, not --rbind), so nothing mounted inside it comes along with
# flags of its own, then made read only.
IFS=: read -r -a binds <<< "$reads"
for d in "${binds[@]}"; do
    if [ -L "$d" ]; then
        cp -P "$d" "$r$d" || fail "cannot link $d"
    elif [ -e "$d" ]; then
        { if [ -d "$d" ]; then mkdir -p "$r$d"; else touch "$r$d"; fi; } \
            && mount -n --bind "$d" "$r$d" \
            && mount -n -o remount,bind,ro "$r$d" || fail "cannot bind $d"
    fi
done
# /usr/local keeps only its programs and libraries, as on macOS: its etc, share, src,
# var and the rest are each covered by an empty, read-only mount.
IFS=: read -r -a keep <<< "$kept"
for e in "$r"/usr/local/* "$r"/usr/local/.[!.]*; do
    { [ -e "$e" ] && [ ! -L "$e" ]; } || continue
    name="${e##*/}"; skip=
    for k in "${keep[@]}"; do
        # shellcheck disable=SC2254  # $k is a glob pattern
        case "$name" in $k) skip=1 ;; esac
    done
    [ -n "$skip" ] && continue
    if [ -d "$e" ]; then
        mount -n -t tmpfs -o ro,size=4k,mode=0555 seal "$e"
    else
        mount -n --bind /dev/null "$e" && mount -n -o remount,bind,ro "$e"
    fi || fail "cannot cover $e"
done
mkdir -p "$r/dev" "$r/proc" "$r/tmp" || fail "cannot make the mount points"
for n in null zero random urandom; do
    { touch "$r/dev/$n" && mount -n --bind "/dev/$n" "$r/dev/$n"; } || fail "cannot bind /dev/$n"
done
{ ln -s /proc/self/fd "$r/dev/fd" && ln -s /proc/self/fd/0 "$r/dev/stdin" \
    && ln -s /proc/self/fd/1 "$r/dev/stdout" && ln -s /proc/self/fd/2 "$r/dev/stderr"; } \
    || fail "cannot link /dev/fd"
mount -n -t proc proc "$r/proc" || fail "cannot mount /proc"
{ mkdir -p "$r$ws" && mount -n --bind /proc/self/fd/3 "$r$ws"; } || fail "cannot bind the workspace"
exec 3<&-
{ cd "$r" && mkdir .old && pivot_root . .old; } || fail "cannot pivot to the new root"
{ cd / && umount -n -l /.old && rmdir /.old; } || fail "cannot detach the old root"
# The new root itself (/, /tmp, the folders above the workspace) is read only, as
# macOS's is: the workspace is the one place a sealed command writes.
mount -n -o remount,bind,ro / || fail "cannot make the new root read only"
# shellcheck disable=SC2086  # $limits is several options
ulimit $limits || fail "cannot set limits"
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do shift; done
[ "$1" = "--" ] || fail "no command"
shift
cd "$ws" || exit 97
exec /usr/bin/unshare --user --mount --map-user="$uid" --map-group="$gid" -- \
    /bin/bash -c "$close_step" refactory-seal /usr/bin/env -i HOME="$ws" TMPDIR="$ws" PATH="$cmd_path" \
    LANG="$locale" LC_ALL="$locale" "$@"
