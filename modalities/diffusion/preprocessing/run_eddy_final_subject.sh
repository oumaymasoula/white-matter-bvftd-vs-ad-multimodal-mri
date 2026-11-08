#!/bin/bash
set -euo pipefail

DATASET=$1
GROUP=$2
SUB=$3

module purge
module load StdEnv/2023
module load fsl/6.0.7.18

PROJECT=${WM_PROJECT_ROOT:?Set WM_PROJECT_ROOT}

OUT="$PROJECT/derivatives/dwi/$DATASET/$GROUP/$SUB"
SYNB0="$PROJECT/derivatives/synb0/$DATASET/$GROUP/$SUB/OUTPUTS"

PREEDDY="$OUT/04_preeddy.nii.gz"
BVEC="$OUT/04_preeddy.bvec"
BVAL="$OUT/04_preeddy.bval"
MASK="$OUT/05_mean_b0_brain_mask.nii.gz"
ACQP="$OUT/acqparams.txt"
INDEX="$OUT/index.txt"
TOPUP="$SYNB0/topup"

echo "======================================"
echo "$DATASET | $GROUP | $SUB"
echo "======================================"

for f in \
"$PREEDDY" \
"$BVEC" \
"$BVAL" \
"$MASK" \
"$ACQP" \
"$INDEX" \
"${TOPUP}_fieldcoef.nii.gz" \
"${TOPUP}_movpar.txt"
do
    [ -s "$f" ] || {
        echo "ERROR: missing $f"
        exit 1
    }
done

echo "===== EDDY + SYNB0/TOPUP ====="

eddy_cuda11.0 \
--imain="$PREEDDY" \
--mask="$MASK" \
--acqp="$ACQP" \
--index="$INDEX" \
--bvecs="$BVEC" \
--bvals="$BVAL" \
--topup="$TOPUP" \
--out="$OUT/06_eddy_gpu" \
--repol \
--cnr_maps \
--residuals

echo "===== POST-EDDY B0 ====="

B0VOLS=$(tr ' ' '\n' < "$BVAL" | \
awk '$1<100 {printf "%s%s",sep,NR-1; sep=","}')

fslselectvols \
-i "$OUT/06_eddy_gpu.nii.gz" \
-o "$OUT/07_posteddy_b0s.nii.gz" \
--vols="$B0VOLS"

fslmaths \
"$OUT/07_posteddy_b0s.nii.gz" \
-Tmean \
"$OUT/07_posteddy_mean_b0.nii.gz"

bet \
"$OUT/07_posteddy_mean_b0.nii.gz" \
"$OUT/07_posteddy_mean_b0_brain.nii.gz" \
-m \
-f 0.25

echo "===== DTI FIT ====="

dtifit \
--data="$OUT/06_eddy_gpu.nii.gz" \
--out="$OUT/08_dtifit" \
--mask="$OUT/07_posteddy_mean_b0_brain_mask.nii.gz" \
--bvecs="$OUT/06_eddy_gpu.eddy_rotated_bvecs" \
--bvals="$BVAL" \
--save_tensor

cp "$OUT/08_dtifit_L1.nii.gz" "$OUT/09_AD.nii.gz"

fslmaths \
"$OUT/08_dtifit_L2.nii.gz" \
-add "$OUT/08_dtifit_L3.nii.gz" \
-div 2 \
"$OUT/09_RD.nii.gz"

echo "===== REQUIRED OUTPUTS ====="

for f in \
"$OUT/06_eddy_gpu.nii.gz" \
"$OUT/06_eddy_gpu.eddy_rotated_bvecs" \
"$OUT/08_dtifit_FA.nii.gz" \
"$OUT/08_dtifit_MD.nii.gz" \
"$OUT/09_AD.nii.gz" \
"$OUT/09_RD.nii.gz"
do
    [ -s "$f" ] || {
        echo "FAIL $f"
        exit 1
    }
    echo "PASS $f"
done

echo "===== DONE ====="
