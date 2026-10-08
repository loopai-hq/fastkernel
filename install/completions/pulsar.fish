# Keep the stable install path so an already-running shell survives upgrades.
set -g __pulsar_completion_source (builtin realpath -s -- (status filename))

function __pulsar_models
    # The helper beside the script's current target, found again each time.
    set -l helper (path dirname -- (path resolve -- $__pulsar_completion_source))/models
    test -x $helper
    and $helper
end

# As in Bash and Zsh: only the value of --model, and only after `pulsar serve`.
function __pulsar_model_value
    set -l words (commandline -opc)
    test "$words[2]" = serve
    and not string match -qr -- '^-[^=]*$' (commandline -ct)
end

complete -c pulsar -f
complete -c pulsar -n 'test (count (commandline -opc)) -eq 1' -a 'serve claude codex opencode hermes pi'
complete -c pulsar -n __pulsar_model_value -l model -x -a '(__pulsar_models)'
