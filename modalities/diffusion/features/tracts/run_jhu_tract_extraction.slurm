#!/bin/bash
#SBATCH --job-name=jhuTract347
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G

set -euo pipefail

module load StdEnv/2023 fsl/6.0.7.18

export TBSSROOT=${WM_TBSS_ROOT:?Set WM_TBSS_ROOT}
export STATS="$TBSSROOT/stats"
export ORDER="$TBSSROOT/non_FA_preparation/non_FA_subject_order.tsv"
export COVERAGE="$TBSSROOT/projection_qc/corrected_subject_tract_coverage.tsv"
export NEGATIVE="$TBSSROOT/non_FA_qc/negative_value_localization/subject_tract_audit/subject_tract_negative_values.tsv"
export DECISION="$TBSSROOT/non_FA_qc/final_non_FA_projection_qc_decision.txt"
export EXTRACT="$TBSSROOT/jhu_tract_metrics"
export ATLAS="$FSLDIR/data/atlases/JHU/JHU-ICBM-tracts-prob-1mm.nii.gz"
export SKELETON="$STATS/mean_FA_skeleton_mask.nii.gz"

"$FSLDIR/bin/python" \
  "${WM_TRACT_EXTRACTOR:?Set WM_TRACT_EXTRACTOR to the absolute repository script path}"
