#!/bin/bash
#SBATCH --job-name=tbssReg347
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-344%20

set -euo pipefail

module purge
module load StdEnv/2023 fsl/6.0.7.18

export OMP_NUM_THREADS=1
export FSLOUTPUTTYPE=NIFTI_GZ

TBSSROOT=${WM_TBSS_ROOT:?Set WM_TBSS_ROOT}
FA="$TBSSROOT/FA"
REGROOT="$TBSSROOT/registration_full"
MANIFEST="$REGROOT/subjects_remaining.tsv"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
LINE=$(sed -n "${LINE_NUMBER}p" "$MANIFEST")
test -n "$LINE"

IFS=$'\t' read -r ARRAY_TASK NAME <<< "$LINE"
test "$ARRAY_TASK" -eq "$SLURM_ARRAY_TASK_ID"

INPUT="$FA/${NAME}.nii.gz"
OUTPUT="$FA/${NAME}_to_target"
WARP="${OUTPUT}_warp.nii.gz"
MSF="${OUTPUT}_warp.msf"
FSLREG_LOG="${OUTPUT}.log"

test -s "$INPUT"
test "$("$FSLDIR/bin/imtest" "$FA/target")" -eq 1

if compgen -G "${OUTPUT}*" >/dev/null; then
    printf 'ERROR: Existing output detected for %s\n' "$NAME"
    exit 1
fi

printf 'Registering: %s\n' "$NAME"

cd "$FA"

"$FSLDIR/bin/fsl_reg" \
    "$NAME" \
    target \
    "${NAME}_to_target" \
    -e \
    -FA

test -s "$WARP"
test -s "$MSF"
test -s "$FSLREG_LOG"
test "$("$FSLDIR/bin/imtest" "$WARP")" -eq 1

# Detect actual failure phrases without matching valid options such as infwhm.
if grep -Eqi \
  '(^|[^[:alpha:]])(error|exception|segmentation fault|aborted|failed)([^[:alpha:]]|$)' \
  "$FSLREG_LOG"; then
    printf 'ERROR: failure phrase found in %s\n' "$FSLREG_LOG"
    grep -Ein \
      '(^|[^[:alpha:]])(error|exception|segmentation fault|aborted|failed)([^[:alpha:]]|$)' \
      "$FSLREG_LOG"
    exit 1
fi

read -r MSF_MEAN MSF_MEDIAN < "$MSF"

test -n "$MSF_MEAN"
test -n "$MSF_MEDIAN"

read -r WARP_MIN WARP_MAX < <(
    "$FSLDIR/bin/fslstats" "$WARP" -R
)

for VALUE in \
    "$MSF_MEAN" "$MSF_MEDIAN" \
    "$WARP_MIN" "$WARP_MAX"
do
    awk -v value="$VALUE" '
    BEGIN {
        lower=tolower(value)
        if (lower=="nan" || lower=="inf" || lower=="-inf")
            exit 1
    }'
done

WARP_DIM1=$("$FSLDIR/bin/fslval" "$WARP" dim1)
WARP_DIM2=$("$FSLDIR/bin/fslval" "$WARP" dim2)
WARP_DIM3=$("$FSLDIR/bin/fslval" "$WARP" dim3)
WARP_DIM4=$("$FSLDIR/bin/fslval" "$WARP" dim4)

printf 'PASS\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$NAME" \
    "$WARP_DIM1" "$WARP_DIM2" "$WARP_DIM3" "$WARP_DIM4" \
    "$WARP_MIN" "$WARP_MAX" \
    "$MSF_MEAN" "$MSF_MEDIAN" \
    > "$REGROOT/status_${SLURM_ARRAY_TASK_ID}.tsv"

printf 'Registration passed: %s\n' "$NAME"
