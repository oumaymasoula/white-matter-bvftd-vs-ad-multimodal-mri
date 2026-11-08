#!/bin/bash

# ============================================================
# DWI STAGE 1 — CPU ONLY
# Import → denoise → degibbs → mean b0 → mask
# → acquisition/index files → integrity QC
# ============================================================

module purge
module load StdEnv/2023
module load mrtrix/3.0.8
module load fsl/6.0.7.7

set -euo pipefail

DATASET=$1
GROUP=$2
SUB=$3

PROJECT=${WM_PROJECT_ROOT:?Set WM_PROJECT_ROOT}

# ------------------------------------------------------------
# Dataset-specific input root
# ------------------------------------------------------------

if [ "$DATASET" = "ADNI" ]; then
    ROOT=${ADNI_BIDS_ROOT:?Set ADNI_BIDS_ROOT}/$GROUP/$SUB
elif [ "$DATASET" = "NIFD" ]; then
    ROOT=${NIFD_BIDS_ROOT:?Set NIFD_BIDS_ROOT}/$GROUP/$SUB
else
    echo "ERROR: unknown dataset: $DATASET"
    exit 1
fi

DWI=$ROOT/dwi/${SUB}_dwi.nii.gz
BVAL=$ROOT/dwi/${SUB}_dwi.bval
BVEC=$ROOT/dwi/${SUB}_dwi.bvec
JSON=$ROOT/dwi/${SUB}_dwi.json

OUT=$PROJECT/derivatives/dwi/$DATASET/$GROUP/$SUB

mkdir -p "$OUT"

DONE=$OUT/STAGE1_CPU_DONE.txt

echo "=================================================="
echo "DWI STAGE 1"
echo "$DATASET | $GROUP | $SUB"
echo "=================================================="
date
hostname

# ------------------------------------------------------------
# Skip already-completed Stage 1
# ------------------------------------------------------------

if [ -s "$DONE" ]; then
    echo "STAGE 1 ALREADY COMPLETE — SKIPPING"
    exit 0
fi

# ------------------------------------------------------------
# Input check
# ------------------------------------------------------------

echo
echo "===== INPUT CHECK ====="

for F in "$DWI" "$BVAL" "$BVEC" "$JSON"; do
    if [ ! -s "$F" ]; then
        echo "ERROR: missing input: $F"
        exit 1
    fi
done

ls -lh "$DWI" "$BVAL" "$BVEC" "$JSON"

# ------------------------------------------------------------
# 1. Import DWI + gradients
# ------------------------------------------------------------

echo
echo "===== MRCONVERT ====="

mrconvert \
    "$DWI" \
    "$OUT/01_raw.mif" \
    -fslgrad "$BVEC" "$BVAL" \
    -force

# ------------------------------------------------------------
# 2. MP-PCA denoising
# ------------------------------------------------------------

echo
echo "===== DENOISING ====="

dwidenoise \
    "$OUT/01_raw.mif" \
    "$OUT/02_denoised.mif" \
    -noise "$OUT/02_noise.mif" \
    -force

# ------------------------------------------------------------
# 3. Gibbs-ringing correction
# ------------------------------------------------------------

echo
echo "===== DEGIBBS ====="

mrdegibbs \
    "$OUT/02_denoised.mif" \
    "$OUT/03_degibbs.mif" \
    -force

# ------------------------------------------------------------
# 4. Export NIfTI + gradients for FSL EDDY
# ------------------------------------------------------------

echo
echo "===== EXPORT PRE-EDDY DATA ====="

mrconvert \
    "$OUT/03_degibbs.mif" \
    "$OUT/04_preeddy.nii.gz" \
    -export_grad_fsl \
    "$OUT/04_preeddy.bvec" \
    "$OUT/04_preeddy.bval" \
    -force

# ------------------------------------------------------------
# 5. Mean b0
# ------------------------------------------------------------

echo
echo "===== MEAN B0 ====="

dwiextract \
    "$OUT/03_degibbs.mif" \
    -bzero - | \
mrmath \
    - \
    mean \
    "$OUT/05_mean_b0.mif" \
    -axis 3 \
    -force

mrconvert \
    "$OUT/05_mean_b0.mif" \
    "$OUT/05_mean_b0.nii.gz" \
    -force

# ------------------------------------------------------------
# 6. Brain mask
# ------------------------------------------------------------

echo
echo "===== B0 BRAIN MASK ====="

bet \
    "$OUT/05_mean_b0.nii.gz" \
    "$OUT/05_mean_b0_brain.nii.gz" \
    -m \
    -f 0.25

MASK=$OUT/05_mean_b0_brain_mask.nii.gz

if [ ! -s "$MASK" ]; then
    echo "ERROR: brain mask not generated"
    exit 1
fi

# ------------------------------------------------------------
# 7. Read acquisition information from BIDS JSON
# ------------------------------------------------------------

echo
echo "===== ACQUISITION PARAMETERS ====="

python - "$JSON" "$OUT/acqparams.txt" <<'PY'
import json
import sys

json_file = sys.argv[1]
outfile = sys.argv[2]

with open(json_file) as f:
    j = json.load(f)

pe = j.get("PhaseEncodingDirection")
ro = j.get("TotalReadoutTime")

if pe is None:
    raise RuntimeError("Missing PhaseEncodingDirection")

if ro is None:
    raise RuntimeError("Missing TotalReadoutTime")

vectors = {
    "i":  ( 1, 0, 0),
    "i-": (-1, 0, 0),
    "j":  ( 0, 1, 0),
    "j-": ( 0,-1, 0),
    "k":  ( 0, 0, 1),
    "k-": ( 0, 0,-1),
}

if pe not in vectors:
    raise RuntimeError(f"Unsupported PhaseEncodingDirection: {pe}")

x, y, z = vectors[pe]

with open(outfile, "w") as f:
    f.write(f"{x} {y} {z} {ro}\n")

print("PhaseEncodingDirection:", pe)
print("TotalReadoutTime:", ro)
print("acqparams:", x, y, z, ro)
PY

# ------------------------------------------------------------
# 8. Create EDDY index.txt
# ------------------------------------------------------------

NVOL=$(fslval "$OUT/04_preeddy.nii.gz" dim4 | tr -d '[:space:]')

python - "$OUT/index.txt" "$NVOL" <<'PY'
import sys

outfile = sys.argv[1]
n = int(sys.argv[2])

with open(outfile, "w") as f:
    f.write(" ".join(["1"] * n) + "\n")

print("index entries:", n)
PY

# ------------------------------------------------------------
# 9. Gradient integrity check
# ------------------------------------------------------------

echo
echo "===== GRADIENT / VOLUME CHECK ====="

NBVAL=$(awk '{for(i=1;i<=NF;i++) n++} END{print n}' \
    "$OUT/04_preeddy.bval")

NBVEC=$(awk 'NR==1 {print NF}' \
    "$OUT/04_preeddy.bvec")

NINDEX=$(awk '{print NF}' \
    "$OUT/index.txt")

echo "DWI volumes   : $NVOL"
echo "bvals         : $NBVAL"
echo "bvec columns  : $NBVEC"
echo "index entries : $NINDEX"

if [ "$NVOL" -ne "$NBVAL" ] || \
   [ "$NVOL" -ne "$NBVEC" ] || \
   [ "$NVOL" -ne "$NINDEX" ]; then

    echo "ERROR: volume / gradient mismatch"
    exit 1
fi

# ------------------------------------------------------------
# 10. Mask quantitative check
# ------------------------------------------------------------

echo
echo "===== BRAIN MASK VOLUME ====="

fslstats "$MASK" -V

# ------------------------------------------------------------
# Completion marker
# ------------------------------------------------------------

echo "STAGE1_CPU_COMPLETE" > "$DONE"

echo
echo "===== STAGE 1 OUTPUT CHECK ====="

ls -lh \
    "$OUT/04_preeddy.nii.gz" \
    "$OUT/04_preeddy.bval" \
    "$OUT/04_preeddy.bvec" \
    "$OUT/05_mean_b0.nii.gz" \
    "$MASK" \
    "$OUT/acqparams.txt" \
    "$OUT/index.txt" \
    "$DONE"

echo
echo "DWI STAGE 1 CPU COMPLETE"
date
