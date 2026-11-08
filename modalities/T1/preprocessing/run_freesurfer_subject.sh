#!/bin/bash

# FreeSurfer environment must be initialized before strict mode
module load StdEnv/2023
module load freesurfer/8.2.0-1
source "$EBROOTFREESURFER/FreeSurferEnv.sh"

set -euo pipefail

DATASET=$1
GROUP=$2
SUB=$3

PROJECT=${WM_PROJECT_ROOT:?Set WM_PROJECT_ROOT}
export FS_LICENSE="${FS_LICENSE:?Set FS_LICENSE to the license file}"

if [ "$DATASET" = "ADNI" ]; then
    ROOT=${ADNI_BIDS_ROOT:?Set ADNI_BIDS_ROOT}/$GROUP/$SUB
elif [ "$DATASET" = "NIFD" ]; then
    ROOT=${NIFD_BIDS_ROOT:?Set NIFD_BIDS_ROOT}/$GROUP/$SUB
else
    echo "ERROR: unknown dataset $DATASET"
    exit 1
fi

T1="$ROOT/anat/${SUB}_T1w.nii.gz"

export SUBJECTS_DIR="$PROJECT/derivatives/freesurfer/$DATASET/$GROUP"
FS="$SUBJECTS_DIR/$SUB"

mkdir -p "$SUBJECTS_DIR"

echo "======================================"
echo "$DATASET | $GROUP | $SUB"
echo "======================================"
echo "T1=$T1"
echo "SUBJECTS_DIR=$SUBJECTS_DIR"
echo "FS=$FS"

[ -s "$T1" ] || {
    echo "ERROR: missing T1: $T1"
    exit 1
}

[ -r "$FS_LICENSE" ] || {
    echo "ERROR: FreeSurfer license not readable"
    exit 1
}

echo
echo "===== FREESURFER VERSION ====="
recon-all -version

# --------------------------------------------------
# CASE 1: subject already completed
# --------------------------------------------------
if [ -f "$FS/scripts/recon-all.log" ] && \
   grep -q "finished without error" "$FS/scripts/recon-all.log"; then

    echo
    echo "===== EXISTING COMPLETED SUBJECT ====="
    echo "FreeSurfer already completed; not rerunning recon-all."

# --------------------------------------------------
# CASE 2: existing but incomplete subject
# --------------------------------------------------
elif [ -d "$FS" ] && [ -f "$FS/mri/orig.mgz" ]; then

    echo
    echo "===== CONTINUING EXISTING SUBJECT ====="
    echo "Existing FreeSurfer directory found."
    echo "Continuing without -i."

    recon-all \
    -s "$SUB" \
    -all \
    -parallel \
    -openmp "${SLURM_CPUS_PER_TASK:-8}"

# --------------------------------------------------
# CASE 3: new subject
# --------------------------------------------------
else

    echo
    echo "===== NEW SUBJECT ====="

    # Remove only an empty/incomplete directory that lacks orig.mgz
    if [ -d "$FS" ]; then
        echo "Removing incomplete subject directory with no orig.mgz:"
        echo "$FS"
        rm -rf "$FS"
    fi

    recon-all \
    -s "$SUB" \
    -i "$T1" \
    -all \
    -parallel \
    -openmp "${SLURM_CPUS_PER_TASK:-8}"
fi

echo
echo "===== FINAL COMPLETION CHECK ====="

grep -q "finished without error" "$FS/scripts/recon-all.log" || {
    echo "ERROR: recon-all completion string missing"
    exit 1
}

for f in \
"$FS/stats/aseg.stats" \
"$FS/stats/lh.aparc.stats" \
"$FS/stats/rh.aparc.stats" \
"$FS/mri/aseg.mgz" \
"$FS/mri/aparc+aseg.mgz" \
"$FS/surf/lh.thickness" \
"$FS/surf/rh.thickness"
do
    if [ -s "$f" ]; then
        echo "PASS $f"
    else
        echo "ERROR missing: $f"
        exit 1
    fi
done

echo
echo "===== FREESURFER PASS ====="
