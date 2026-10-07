#!/bin/sh
# Modified by meowkernels.
# Build dist/fastkernel-VERSION-macos-arm64.tar.gz: a prebuilt package that runs without Xcode.
# Same layout as Splash's release package; release.json marks it as packaged (install/paths.py).
# Usage: make all && install/package.sh [VERSION]
set -eu
cd "$(dirname -- "$0")/.."
version=${1:-1.1.1}
test -x build/splash && test -f build/splash.metallib || { echo "error: run 'make all' first" >&2; exit 1; }

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
pkg=$stage/fastkernel
mkdir -p "$pkg/engine" dist
git archive HEAD install server data/head-ranked.u32 LICENSE NOTICE THIRD_PARTY_NOTICES | tar -x -C "$pkg"
cp build/splash build/splash.metallib "$pkg/engine/"
cat > "$pkg/release.json" <<EOF
{
 "version": "fastkernel $version",
 "binary_sha256": "$(shasum -a 256 build/splash | cut -d' ' -f1)",
 "metallib_sha256": "$(shasum -a 256 build/splash.metallib | cut -d' ' -f1)"
}
EOF

cat > "$pkg/splash" <<'EOF'
#!/bin/sh
# fastkernel launcher: sets up python/ on first run, then starts the Splash launcher.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
test "$(uname -m)" = arm64 || { echo "fastkernel needs an Apple silicon Mac." >&2; exit 1; }
sw_vers -productVersion | awk -F. '{exit !($1 > 26 || ($1 == 26 && $2 >= 4))}' \
    || { echo "fastkernel needs macOS 26.4 or later." >&2; exit 1; }
if /usr/bin/xattr -p com.apple.quarantine "$root/engine/splash" >/dev/null 2>&1; then
    echo "macOS quarantined this download. Run this once, then try again:" >&2
    echo "  /usr/bin/xattr -dr com.apple.quarantine \"$root\"" >&2
    exit 1
fi
if ! test -f "$root/python/.requirements-installed"; then
    # Try each Python until one can build the environment (one may lack network access).
    for candidate in python3.13 python3 python3.12 python3.14; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        base=$("$candidate" -c 'import sys; print(getattr(sys, "_base_executable", sys.executable))') || continue
        "$base" -c 'import sys; raise SystemExit(not ((3, 12) <= sys.version_info[:2] < (3, 15)))' 2>/dev/null || continue
        echo "First run: installing Python packages into $root/python with $base" >&2
        "$base" -m venv --clear "$root/python" \
            && "$root/python/bin/python3" -m pip install -q --only-binary=:all: -r "$root/install/requirements.txt" \
            && touch "$root/python/.requirements-installed" && break
    done
    test -f "$root/python/.requirements-installed" \
        || { echo "fastkernel needs Python 3.12, 3.13, or 3.14 and internet access on its first run." >&2; exit 1; }
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$root/python/bin/python3" -u "$root/install/launcher.py" "$@"
EOF
chmod 0755 "$pkg/splash"

out=dist/fastkernel-$version-macos-arm64.tar.gz
tar -czf "$out" -C "$stage" fastkernel
echo "$out"
