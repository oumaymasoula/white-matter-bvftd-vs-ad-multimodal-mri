#!/bin/bash

set -euo pipefail

DATASET=$1
GROUP=$2
SUB=$3

module purge
module load StdEnv/2023
module load apptainer/1.4.5

HOSTSCRATCH=${WM_SCRATCH_ROOT:?Set WM_SCRATCH_ROOT}
PROJECT=$HOSTSCRATCH/WM_Project
CONTAINER=$HOSTSCRATCH/containers/mars-wmh-nnunet.sif

if [ "$DATASET" = "ADNI" ]; then
    ROOT=$HOSTSCRATCH/ADNI_BIDS/$GROUP/$SUB
elif [ "$DATASET" = "NIFD" ]; then
    ROOT=$HOSTSCRATCH/NIFD_BIDS/$GROUP/$SUB
else
    echo "ERROR: unknown dataset: $DATASET"
    exit 1
fi

T1=$ROOT/anat/${SUB}_T1w.nii.gz
FLAIR=$ROOT/anat/${SUB}_FLAIR.nii.gz

OUT=$PROJECT/derivatives/wmh/$DATASET/$GROUP/$SUB
mkdir -p "$OUT"

FINAL=$OUT/${SUB}_WMH.nii.gz

echo "======================================"
echo "$DATASET | $GROUP | $SUB"
echo "======================================"

if [ -s "$FINAL" ]; then
    echo "OUTPUT ALREADY EXISTS — SKIPPING"
    exit 0
fi

echo "===== INPUT CHECK ====="
ls -lh "$T1" "$FLAIR"

echo
echo "===== MARS-WMH ====="

apptainer run \
    --nv \
    --bind "$HOSTSCRATCH:/work" \
    "$CONTAINER" \
    --flair "/work/${FLAIR#$HOSTSCRATCH/}" \
    --t1 "/work/${T1#$HOSTSCRATCH/}" \
    -o "/work/${FINAL#$HOSTSCRATCH/}" \
    --saveStatistics

echo
echo "===== OUTPUT ====="

ls -lh "$FINAL"

echo
echo "WMH COMPLETE"
