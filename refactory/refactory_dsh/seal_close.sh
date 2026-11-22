# The last step before a sealed command (`refactory_dsh.seal`), in every namespace it
# runs in: close every descriptor above stderr, check that none is left (a command that
# would start with one does not start), then run the arguments. Kept in a file and read
# by seal_linux.sh inside the namespaces, so nothing that rewrites the runner's
# arguments (systemd-run) can turn the check off.
# Descriptor 98 is the runner's start pipe (`seal_reaper.START_FD`): kept until every other
# one is closed and checked, then told the command is starting and closed in turn, so an
# exit after this is the command's own and the command never holds it.
for fd in /dev/fd/*; do
    n="${fd##*/}"
    [ "$n" -gt 2 ] 2>/dev/null && [ "$n" != 98 ] && eval "exec $n>&-" 2>/dev/null
done
for fd in /dev/fd/*; do
    n="${fd##*/}"
    if [ "$n" -gt 2 ] 2>/dev/null && [ "$n" != 98 ] && [ -e "$fd" ]; then
        echo "refactory-seal: descriptor $n is still open" >&2
        exit 97
    fi
done
if [ -e /dev/fd/98 ]; then
    { printf s >&98; } 2>/dev/null
    exec 98>&-
    if [ -e /dev/fd/98 ]; then
        echo "refactory-seal: descriptor 98 is still open" >&2
        exit 97
    fi
fi
exec "$@"
