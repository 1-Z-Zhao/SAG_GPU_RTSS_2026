#!/bin/bash
# Build and optionally run the SAG solver
#
# Usage:
#   ./run.sh --build-only
#   ./run.sh --no-clean -m 4 path/to/jobs.csv path/to/jobsprec.csv

set -e
cd "$(dirname "$0")"

CLEAN=1
RUN=0
M=4
ARGS=()

while [ "$#" -gt 0 ]; do
    arg="$1"
    case "$arg" in
        --no-clean)
            CLEAN=0
            ;;
        --build-only)
            RUN=0
            ;;
        -m)
            shift
            M="$1"
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
    shift
done

if [ "${#ARGS[@]}" -eq 2 ]; then
    RUN=1
elif [ "${#ARGS[@]}" -ne 0 ]; then
    echo "Usage: ./run.sh [--no-clean] [--build-only] [-m CORES] <jobs.csv> <jobsprec.csv>" >&2
    exit 2
fi

# --- Clean ---
if [ "$CLEAN" -eq 1 ]; then
    echo "=== Cleaning build directory ==="
    rm -rf build
fi

# --- Build ---
echo "=== Building ==="
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
cmake --build . -j "$JOBS" 2>&1 | tail -3
cd ..
echo ""

if [ "$RUN" -eq 0 ]; then
    echo "Build complete."
    exit 0
fi

# --- Run provided taskset ---
echo "###########################################################"
echo " Provided taskset  $(date '+%b %d, %Y')"
echo "###########################################################"
echo ""
./build/expand_test -m "$M" "${ARGS[0]}" "${ARGS[1]}"
echo ""

echo "=== Done ==="
