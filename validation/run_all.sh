#!/usr/bin/env bash
# Run all benchmark run.py files in order and report results.
set -uo pipefail

PYTHON="/opt/anaconda3/envs/helix/bin/python"
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)/benchmarks"

pass=0
fail=0
skip=0
total=0
declare -a results=()

for dir in "$BENCH_DIR"/*/; do
    [ -d "$dir" ] || continue
    name="$(basename "$dir")"
    run_py="$dir/run.py"
    total=$((total + 1))

    if [ ! -f "$run_py" ]; then
        echo "SKIP  $name (no run.py)"
        skip=$((skip + 1))
        results+=("SKIP  $name")
        continue
    fi

    output=$("$PYTHON" "$run_py" 2>/dev/null)
    rc=$?

    if [ $rc -eq 0 ]; then
        status=$(echo "$output" | "$PYTHON" -c "import sys,json; print(json.load(sys.stdin).get('status','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
    else
        status="FAIL"
    fi

    case "$status" in
        PASS|SKIP)
            echo "$status  $name"
            if [ "$status" = "PASS" ]; then
                pass=$((pass + 1))
            else
                skip=$((skip + 1))
            fi
            results+=("$status  $name")
            ;;
        *)
            echo "FAIL  $name"
            fail=$((fail + 1))
            results+=("FAIL  $name")
            if [ -n "$output" ]; then
                echo "      $(echo "$output" | head -3)"
            fi
            ;;
    esac
done

echo ""
echo "========================================="
echo "Summary: $total total, $pass PASS, $fail FAIL, $skip SKIP"
echo "========================================="
for r in "${results[@]}"; do
    echo "  $r"
done

if [ $fail -gt 0 ]; then
    exit 1
fi
exit 0
