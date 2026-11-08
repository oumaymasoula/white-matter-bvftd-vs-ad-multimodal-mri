#!/bin/bash
#SBATCH --job-name=tbssNonFA347
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

module --force purge
module load StdEnv/2023
module load fsl/6.0.7.18

TBSSROOT=${WM_TBSS_ROOT:?Set WM_TBSS_ROOT}
STATS="$TBSSROOT/stats"

cd "$TBSSROOT"

for METRIC in MD AD RD; do
    printf '\n===== PROJECTING %s =====\n' "$METRIC"

    for OUTPUT in \
      "$STATS/all_${METRIC}.nii.gz" \
      "$STATS/all_${METRIC}_skeletonised.nii.gz"
    do
        test ! -e "$OUTPUT"
    done

    tbss_non_FA "$METRIC"

    test -s "$STATS/all_${METRIC}.nii.gz"
    test -s "$STATS/all_${METRIC}_skeletonised.nii.gz"

    DIM1=$(
        "$FSLDIR/bin/fslval" \
          "$STATS/all_${METRIC}_skeletonised.nii.gz" dim1
    )
    DIM2=$(
        "$FSLDIR/bin/fslval" \
          "$STATS/all_${METRIC}_skeletonised.nii.gz" dim2
    )
    DIM3=$(
        "$FSLDIR/bin/fslval" \
          "$STATS/all_${METRIC}_skeletonised.nii.gz" dim3
    )
    DIM4=$(
        "$FSLDIR/bin/fslval" \
          "$STATS/all_${METRIC}_skeletonised.nii.gz" dim4
    )
    RANGE=$(
        "$FSLDIR/bin/fslstats" \
          "$STATS/all_${METRIC}_skeletonised.nii.gz" -R
    )

    test "$DIM1" -eq 182
    test "$DIM2" -eq 218
    test "$DIM3" -eq 182
    test "$DIM4" -eq 347

    printf '%s projected dimensions: %s x %s x %s x %s\n' \
      "$METRIC" "$DIM1" "$DIM2" "$DIM3" "$DIM4"

    printf '%s projected range: %s\n' "$METRIC" "$RANGE"
done

cat > "$TBSSROOT/non_FA_projection_status.txt" <<EOF
TBSS NON-FA PROJECTION COMPLETE

Subjects: 347
MD projected: YES
AD projected: YES
RD projected: YES
Subject order: task-prefixed manifest order
Skeleton threshold inherited from FA: 0.20

Next:
- Validate finite values and subject order.
- Perform numerical and visual non-FA projection QC.
- Extract tract summaries only after projection QC passes.
EOF

cat "$TBSSROOT/non_FA_projection_status.txt"
