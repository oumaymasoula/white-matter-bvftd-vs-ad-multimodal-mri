#!/bin/bash
#SBATCH --job-name=tbssPrestats347
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G

set -euo pipefail

module purge
module load StdEnv/2023
module load fsl/6.0.7.18

TBSSROOT=${WM_TBSS_ROOT:?Set WM_TBSS_ROOT}
STATS="$TBSSROOT/stats"

cd "$TBSSROOT"

tbss_4_prestats 0.2

for OUTPUT in \
  "$STATS/all_FA_skeletonised.nii.gz" \
  "$STATS/mean_FA_skeleton_mask.nii.gz" \
  "$STATS/mean_FA_skeleton_mask_dst.nii.gz"
do
    test -s "$OUTPUT"
done

test "$("$FSLDIR/bin/imtest" \
  "$STATS/all_FA_skeletonised")" -eq 1

DIM4=$("$FSLDIR/bin/fslval" \
  "$STATS/all_FA_skeletonised.nii.gz" dim4)

test "$DIM4" -eq 347

MASK_VOXELS=$(
  "$FSLDIR/bin/fslstats" \
    "$STATS/mean_FA_skeleton_mask.nii.gz" -V |
  awk '{print $1}'
)

test "$MASK_VOXELS" -gt 0

RANGE=$(
  "$FSLDIR/bin/fslstats" \
    "$STATS/all_FA_skeletonised.nii.gz" -R
)

cat > "$TBSSROOT/tbss_4_status.txt" <<EOF
TBSS STAGE 4 PRESTATS COMPLETE

Skeleton threshold: 0.20
Subjects projected: $DIM4
Skeleton-mask voxels: $MASK_VOXELS
Skeletonised FA range: $RANGE

Next:
- Run projection QC.
- Quantify subject-by-tract skeleton coverage.
- Do not begin final statistical inference before coverage QC.
EOF

cat "$TBSSROOT/tbss_4_status.txt"
