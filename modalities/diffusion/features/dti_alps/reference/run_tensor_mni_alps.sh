#!/bin/bash
set -euo pipefail

DATASET=$1
GROUP=$2
SUB=$3

module load StdEnv/2023
module load ants/2.6.5
module load fsl/6.0.7.7

PROJECT=${WM_PROJECT_ROOT:?Set WM_PROJECT_ROOT}
ALPSREPO=${ALPS_REPO_ROOT:?Set ALPS_REPO_ROOT}

DWI="$PROJECT/derivatives/dwi_pilot/$DATASET/$GROUP/$SUB"
T1="$PROJECT/derivatives/t1_pilot/$DATASET/$GROUP/$SUB"
ALPSOUT="$PROJECT/derivatives/alps_pilot/$DATASET/$GROUP/$SUB"

MNI="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
MNIMASK="$FSLDIR/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"

mkdir -p "$ALPSOUT"

echo "=============================================="
echo "$DATASET | $GROUP | $SUB"
echo "=============================================="

echo
echo "===== REQUIRED INPUT CHECK ====="

REQ=(
"$DWI/08_dtifit_tensor.nii.gz"
"$DWI/10_DWI_to_T1_0GenericAffine.mat"
"$T1/${SUB}_T1_to_MNI_0GenericAffine.mat"
"$T1/${SUB}_T1_to_MNI_1Warp.nii.gz"
"$ALPSREPO/ROIs_JHU_ALPS/L_SCR.nii.gz"
"$ALPSREPO/ROIs_JHU_ALPS/R_SCR.nii.gz"
"$ALPSREPO/ROIs_JHU_ALPS/L_SLF.nii.gz"
"$ALPSREPO/ROIs_JHU_ALPS/R_SLF.nii.gz"
)

for f in "${REQ[@]}"; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR missing: $f"
        exit 1
    fi
    echo "PASS $f"
done

echo
echo "===== FSL TENSOR -> ITK ====="

ImageMath \
3 \
"$DWI/13_tensor_ITK.nii.gz" \
FSLTensorToITK \
"$DWI/08_dtifit_tensor.nii.gz"

echo
echo "===== NATIVE ITK TENSOR ====="

ls -lh "$DWI/13_tensor_ITK.nii.gz"

fslinfo "$DWI/13_tensor_ITK.nii.gz" | \
grep -E 'dim[1234]|pixdim[123]'

echo
echo "===== TENSOR-AWARE DWI -> T1 -> MNI ====="

antsApplyTransforms \
-d 3 \
-e 2 \
-i "$DWI/13_tensor_ITK.nii.gz" \
-r "$MNI" \
-o "$DWI/18_tensor_ITK_MNI.nii.gz" \
-t "$T1/${SUB}_T1_to_MNI_1Warp.nii.gz" \
-t "$T1/${SUB}_T1_to_MNI_0GenericAffine.mat" \
-t "$DWI/10_DWI_to_T1_0GenericAffine.mat" \
-n Linear \
-f 0.0007

echo
echo "===== MNI ITK TENSOR ====="

ls -lh "$DWI/18_tensor_ITK_MNI.nii.gz"

fslinfo "$DWI/18_tensor_ITK_MNI.nii.gz" | \
grep -E 'dim[1234]|pixdim[123]'

echo
echo "===== ITK TENSOR -> FSL ====="

ImageMath \
3 \
"$DWI/19_tensor_FSL_MNI.nii.gz" \
ITKTensorToFSL \
"$DWI/18_tensor_ITK_MNI.nii.gz"

echo
echo "===== FSL MNI TENSOR ====="

fslinfo "$DWI/19_tensor_FSL_MNI.nii.gz" | \
grep -E 'dim[1234]|pixdim[123]'

echo
echo "===== EXTRACT Dxx / Dyy / Dzz ====="

fslroi \
"$DWI/19_tensor_FSL_MNI.nii.gz" \
"$DWI/20_Dxx_MNI.nii.gz" \
0 1

fslroi \
"$DWI/19_tensor_FSL_MNI.nii.gz" \
"$DWI/20_Dyy_MNI.nii.gz" \
3 1

fslroi \
"$DWI/19_tensor_FSL_MNI.nii.gz" \
"$DWI/20_Dzz_MNI.nii.gz" \
5 1

echo
echo "===== MNI TENSOR COMPONENT RANGES ====="

for f in Dxx Dyy Dzz; do

    echo "$f:"

    fslstats \
    "$DWI/20_${f}_MNI.nii.gz" \
    -k "$MNIMASK" \
    -P 1 -P 50 -P 99

done

echo
echo "===== ALPS ROIs ====="

L_SCR="$ALPSREPO/ROIs_JHU_ALPS/L_SCR.nii.gz"
R_SCR="$ALPSREPO/ROIs_JHU_ALPS/R_SCR.nii.gz"
L_SLF="$ALPSREPO/ROIs_JHU_ALPS/L_SLF.nii.gz"
R_SLF="$ALPSREPO/ROIs_JHU_ALPS/R_SLF.nii.gz"

DXX="$DWI/20_Dxx_MNI.nii.gz"
DYY="$DWI/20_Dyy_MNI.nii.gz"
DZZ="$DWI/20_Dzz_MNI.nii.gz"

x_proj_L=$(fslstats "$DXX" -k "$L_SCR" -M)
x_assoc_L=$(fslstats "$DXX" -k "$L_SLF" -M)

y_proj_L=$(fslstats "$DYY" -k "$L_SCR" -M)
z_assoc_L=$(fslstats "$DZZ" -k "$L_SLF" -M)

x_proj_R=$(fslstats "$DXX" -k "$R_SCR" -M)
x_assoc_R=$(fslstats "$DXX" -k "$R_SLF" -M)

y_proj_R=$(fslstats "$DYY" -k "$R_SCR" -M)
z_assoc_R=$(fslstats "$DZZ" -k "$R_SLF" -M)

python - <<PY
from pathlib import Path

x_proj_L=float("$x_proj_L")
x_assoc_L=float("$x_assoc_L")
y_proj_L=float("$y_proj_L")
z_assoc_L=float("$z_assoc_L")

x_proj_R=float("$x_proj_R")
x_assoc_R=float("$x_assoc_R")
y_proj_R=float("$y_proj_R")
z_assoc_R=float("$z_assoc_R")

alps_L=(x_proj_L+x_assoc_L)/(y_proj_L+z_assoc_L)
alps_R=(x_proj_R+x_assoc_R)/(y_proj_R+z_assoc_R)
alps_mean=(alps_L+alps_R)/2

print("===== ALPS ROI MEANS =====")
print("x_proj_L =", round(x_proj_L,9))
print("x_assoc_L =", round(x_assoc_L,9))
print("y_proj_L =", round(y_proj_L,9))
print("z_assoc_L =", round(z_assoc_L,9))
print("x_proj_R =", round(x_proj_R,9))
print("x_assoc_R =", round(x_assoc_R,9))
print("y_proj_R =", round(y_proj_R,9))
print("z_assoc_R =", round(z_assoc_R,9))

print()
print("ALPS_L =", round(alps_L,6))
print("ALPS_R =", round(alps_R,6))
print("ALPS_mean =", round(alps_mean,6))

out=Path("$ALPSOUT/alps_summary.tsv")

out.write_text(
    "metric\\tvalue\\n"
    f"ALPS_L\\t{alps_L:.6f}\\n"
    f"ALPS_R\\t{alps_R:.6f}\\n"
    f"ALPS_MEAN\\t{alps_mean:.6f}\\n"
)

print()
print("Saved:",out)
PY

echo
echo "===== FINAL FILES ====="

ls -lh \
"$DWI/18_tensor_ITK_MNI.nii.gz" \
"$DWI/19_tensor_FSL_MNI.nii.gz" \
"$DWI/20_Dxx_MNI.nii.gz" \
"$DWI/20_Dyy_MNI.nii.gz" \
"$DWI/20_Dzz_MNI.nii.gz" \
"$ALPSOUT/alps_summary.tsv"

echo
echo "===== DONE ====="
