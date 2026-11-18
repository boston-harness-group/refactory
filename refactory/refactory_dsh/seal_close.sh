# The last step before a sealed command (`refactory_dsh.seal`), in every namespace it
# runs in: close every descriptor above stderr, check that none is left (a command that
# would start with one does not start), then run the arguments. Kept in a file and read
# by seal_linux.sh inside the namespaces, so nothing that rewrites the runner's
# arguments (systemd-run) can turn the check off.
for fd in /dev/fd/*; do
    n="${fd##*/}"
    [ "$n" -gt 2 ] 2>/dev/null && eval "exec $n>&-" 2>/dev/null
done
for fd in /dev/fd/*; do
    n="${fd##*/}"
    if [ "$n" -gt 2 ] 2>/dev/null && [ -e "$fd" ]; then
        echo "refactory-seal: descriptor $n is still open" >&2
        exit 97
    fi
done
exec "$@"
