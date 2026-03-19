#!/bin/bash
#==============================================================================#
#             Timing comparison: reader backends + pipeline modes               #
#==============================================================================#
#
# Tests all combinations of:
#   --reader=pycall  vs  --reader=native
#   sequential       vs  --pipeline
#   with writeback   vs  --skip-writeback
#
# USAGE:
#   bash bin/timing_test.sh <sources.json> <ms1> <ms2> [ms3] [ms4] ...
#
# The script needs at least 2 MS files for pipeline testing.
# It runs 6 configurations, each processing all MS files.
#
# REQUIREMENTS:
#   - JULIA must be set (or julia in PATH)
#   - python3 with casacore + numpy in PATH
#   - GPU available
#
# EXAMPLE (on server):
#   export JULIA=/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin/julia
#   export JULIA_DEPOT_PATH="/tmp/julia_nkosogor_dev:/opt/devel/nkosogor/nkosogor/julia_depot"
#   cd /lustre/nkosogor/ttcalx_imager/TTCalX_imager
#
#   # Copy a few MS files for testing
#   MS_DIR=/lustre/pipeline/night-time/03h/
#   MS1=${MS_DIR}/20240524_030004_73MHz_ch0to5.ms
#   MS2=${MS_DIR}/20240524_030603_73MHz_ch0to5.ms
#   MS3=${MS_DIR}/20240524_031203_73MHz_ch0to5.ms
#   MS4=${MS_DIR}/20240524_031803_73MHz_ch0to5.ms
#
#   bash bin/timing_test.sh /home/pipeline/sources.json $MS1 $MS2 $MS3 $MS4
#==============================================================================#

set -e

JULIA="${JULIA:-julia}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/peel_and_image_gpu.jl"

# Common flags for all runs
COMMON_FLAGS="--skip-before --fits-only --verbose --column=CORRECTED_DATA"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <sources.json> <ms1> <ms2> [ms3] ..."
    echo "Need at least 2 MS files for pipeline testing."
    exit 1
fi

SOURCES="$1"
shift
MS_FILES=("$@")
NMS=${#MS_FILES[@]}

echo "============================================================"
echo "  TTCalX Timing Comparison"
echo "============================================================"
echo "Julia:    $JULIA"
echo "Sources:  $SOURCES"
echo "MS files: $NMS"
for f in "${MS_FILES[@]}"; do
    echo "  $(basename "$f")"
done
echo ""

# Create output directory for timing results
OUTDIR="timing_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
echo "Output dir: $OUTDIR"
echo ""

# Define test configurations
declare -a CONFIG_NAMES
declare -a CONFIG_FLAGS

# Config A: PyCall reader, sequential, with writeback
CONFIG_NAMES+=("A_pycall_seq_write")
CONFIG_FLAGS+=("--reader=pycall")

# Config B: PyCall reader, sequential, skip writeback
CONFIG_NAMES+=("B_pycall_seq_nowrite")
CONFIG_FLAGS+=("--reader=pycall --skip-writeback")

# Config C: Native reader, sequential, with writeback
CONFIG_NAMES+=("C_native_seq_write")
CONFIG_FLAGS+=("--reader=native")

# Config D: Native reader, sequential, skip writeback
CONFIG_NAMES+=("D_native_seq_nowrite")
CONFIG_FLAGS+=("--reader=native --skip-writeback")

# Config E: Native reader, pipeline, with writeback
CONFIG_NAMES+=("E_native_pipe_write")
CONFIG_FLAGS+=("--reader=native --pipeline")

# Config F: Native reader, pipeline, skip writeback
CONFIG_NAMES+=("F_native_pipe_nowrite")
CONFIG_FLAGS+=("--reader=native --pipeline --skip-writeback")

NCONFIGS=${#CONFIG_NAMES[@]}

echo "Configurations to test: $NCONFIGS"
for i in $(seq 0 $((NCONFIGS-1))); do
    echo "  ${CONFIG_NAMES[$i]}: ${CONFIG_FLAGS[$i]}"
done
echo ""

# Run each configuration
for i in $(seq 0 $((NCONFIGS-1))); do
    NAME="${CONFIG_NAMES[$i]}"
    FLAGS="${CONFIG_FLAGS[$i]}"
    LOGFILE="$OUTDIR/${NAME}.log"
    
    echo "============================================================"
    echo "  [$((i+1))/$NCONFIGS] $NAME"
    echo "  Flags: $FLAGS"
    echo "============================================================"
    
    # Build the command
    CMD="$JULIA $PIPELINE_SCRIPT peel $COMMON_FLAGS $FLAGS --output=$OUTDIR/${NAME} $SOURCES ${MS_FILES[*]}"
    
    echo "Command: $CMD"
    echo ""
    
    # Run and capture wall-clock time
    START_TIME=$(date +%s%N)
    
    eval "$CMD" 2>&1 | tee "$LOGFILE"
    EXIT_CODE=${PIPESTATUS[0]}
    
    END_TIME=$(date +%s%N)
    WALL_MS=$(( (END_TIME - START_TIME) / 1000000 ))
    WALL_SEC=$(echo "scale=2; $WALL_MS / 1000" | bc)
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  *** FAILED (exit code $EXIT_CODE) ***"
    fi
    
    echo ""
    echo "  Wall-clock: ${WALL_SEC}s"
    echo "  Log: $LOGFILE"
    echo ""
    
    # Record summary
    echo "${NAME}  wall=${WALL_SEC}s  exit=$EXIT_CODE" >> "$OUTDIR/summary.txt"
done

# Print final summary
echo ""
echo "============================================================"
echo "  TIMING SUMMARY"
echo "============================================================"
echo ""
printf "%-30s  %10s  %s\n" "Configuration" "Wall Time" "Status"
printf "%-30s  %10s  %s\n" "------------------------------" "----------" "------"
while IFS= read -r line; do
    NAME=$(echo "$line" | awk '{print $1}')
    WALL=$(echo "$line" | awk -F'wall=' '{print $2}' | awk '{print $1}')
    EXIT=$(echo "$line" | awk -F'exit=' '{print $2}')
    STATUS="OK"
    [ "$EXIT" != "0" ] && STATUS="FAIL"
    printf "%-30s  %10s  %s\n" "$NAME" "$WALL" "$STATUS"
done < "$OUTDIR/summary.txt"
echo ""
echo "Detailed logs in: $OUTDIR/"
echo ""

# Extract per-MS averages from logs
echo "Per-MS averages (excluding JIT, from Julia output):"
echo ""
for i in $(seq 0 $((NCONFIGS-1))); do
    NAME="${CONFIG_NAMES[$i]}"
    LOGFILE="$OUTDIR/${NAME}.log"
    AVG=$(grep "Average (excl. JIT)" "$LOGFILE" 2>/dev/null | grep -oP '[\d.]+(?= seconds)' || echo "N/A")
    printf "  %-30s  avg=%s s/MS\n" "$NAME" "$AVG"
done
echo ""
echo "Done!"
