#!/bin/bash
#SBATCH --job-name=tbssPre347
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G

set -euo pipefail

module purge
module load StdEnv/2023 fsl/6.0.7.18

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLBACKEND=Agg

TBSSROOT=${WM_TBSS_ROOT:?Set WM_TBSS_ROOT}
cd "$TBSSROOT"

test "$(find . -maxdepth 1 -type f -name '*_FA.nii.gz' | wc -l)" -eq 347

tbss_1_preproc ./*_FA.nii.gz

test -d "$TBSSROOT/FA"
test -d "$TBSSROOT/origdata"

ORIGINALS=$(find "$TBSSROOT/origdata" -maxdepth 1 -type f \
    -name '*_FA.nii.gz' | wc -l)

PREPROCESSED=$(find "$TBSSROOT/FA" -maxdepth 1 -type f \
    -name '*_FA.nii.gz' | wc -l)

printf 'Original FA images: %s\n' "$ORIGINALS"
printf 'Preprocessed FA images: %s\n' "$PREPROCESSED"

test "$ORIGINALS" -eq 347
test "$PREPROCESSED" -eq 347

printf '347 TBSS FA INPUTS PREPROCESSED — VISUAL CHECK REQUIRED\n' \
    > "$TBSSROOT/tbss_1_status.txt"

printf '\nTBSS stage 1 completed.\n'
printf 'No production diffusion or tensor image was modified.\n'
