#!/bin/bash
#SBATCH --job-name=tbssPost347
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

module purge
module load StdEnv/2023 fsl/6.0.7.18

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export FSLOUTPUTTYPE=NIFTI_GZ

TBSSROOT=${WM_TBSS_ROOT:?Set WM_TBSS_ROOT}
REGQC="$TBSSROOT/registration_qc"
FA="$TBSSROOT/FA"

test -s "$REGQC/numerical_status.txt"

test "$(find "$FA" -maxdepth 1 -type f \
  -name 't[0-9][0-9][0-9][0-9]__*_FA_FA.nii.gz' | wc -l)" -eq 347

test "$(find "$FA" -maxdepth 1 -type f \
  -name 't[0-9][0-9][0-9][0-9]__*_FA_FA_to_target_warp.nii.gz' | wc -l)" -eq 347

cd "$TBSSROOT"

"$FSLDIR/bin/tbss_3_postreg" -S

test -s "$TBSSROOT/stats/all_FA.nii.gz"
test -s "$TBSSROOT/stats/mean_FA.nii.gz"
test -s "$TBSSROOT/stats/mean_FA_skeleton.nii.gz"

SUBJECTS=$(
    "$FSLDIR/bin/fslval" \
      "$TBSSROOT/stats/all_FA.nii.gz" dim4
)

DIM1=$(
    "$FSLDIR/bin/fslval" \
      "$TBSSROOT/stats/all_FA.nii.gz" dim1
)
DIM2=$(
    "$FSLDIR/bin/fslval" \
      "$TBSSROOT/stats/all_FA.nii.gz" dim2
)
DIM3=$(
    "$FSLDIR/bin/fslval" \
      "$TBSSROOT/stats/all_FA.nii.gz" dim3
)

test "$SUBJECTS" -eq 347
test "$DIM1" -eq 182
test "$DIM2" -eq 218
test "$DIM3" -eq 182

read -r ALL_MIN ALL_MAX < <(
    "$FSLDIR/bin/fslstats" \
      "$TBSSROOT/stats/all_FA.nii.gz" -R
)

awk -v minimum="$ALL_MIN" -v maximum="$ALL_MAX" '
BEGIN {
    if (minimum < -0.0001 || maximum <= 0 || maximum > 1.1)
        exit 1
}'

cat > "$REGQC/postreg_status.txt" <<EOF
TBSS POST-REGISTRATION PROCESSING COMPLETE

Subjects in all_FA: $SUBJECTS
Standard-space geometry: ${DIM1}x${DIM2}x${DIM3}
all_FA range: $ALL_MIN to $ALL_MAX
Mean FA created: YES
Study-specific mean FA skeleton created: YES

Final visual registration review required.
Skeleton threshold has not yet been accepted.
EOF

printf '\n===== POST-REGISTRATION STATUS =====\n'
cat "$REGQC/postreg_status.txt"
