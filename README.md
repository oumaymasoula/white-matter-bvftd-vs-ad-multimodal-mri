# White Matter MRI in bvFTD, AD, and MCI

Analysis code for *Anterior White Matter Diffusion Changes in Behavioral Variant Frontotemporal Dementia Versus Alzheimer Disease: An Exploratory Cross-Cohort MRI Study*.

The study compares white matter diffusion in behavioral variant frontotemporal dementia and Alzheimer disease using cohort-specific controls, with mild cognitive impairment included in secondary comparisons, alongside DTI-ALPS, white matter hyperintensity burden, and structural MRI. The cross-cohort comparisons are exploratory: diagnosis and acquisition cannot be separated in this design.

The analysis starts from authorized feature tables. This code package does not include participant data or a complete workflow from raw images.

## Repository layout

| Path | Purpose |
|---|---|
| `modalities/diffusion/preprocessing/` | DWI preparation and final EDDY subject scripts |
| `modalities/diffusion/features/tracts/` | TBSS and JHU tract feature extraction |
| `modalities/diffusion/features/dti_alps/` | DTI-ALPS method and reference script |
| `modalities/T1/preprocessing/` | FreeSurfer reconstruction |
| `modalities/T1/features/` | Structural feature extraction |
| `modalities/FLAIR/preprocessing/` | FLAIR/T1 white matter hyperintensity segmentation |
| `analysis/` | Diffusion, structural, clinical, multimodal and plotting code |
| `workflow/` | Primary-model entry point |
| `provenance/` and `environment/` | Source hashes, input fingerprints and software environment |

See [workflow scope and configuration](docs/modality_layout.md) before running the preprocessing scripts.

## Run the primary models

Install the packages in `environment/primary_requirements.txt` in an isolated Python environment. Obtain the exact five authorized input tables described in [Inputs and outputs](docs/inputs_outputs.md), and verify them against `provenance/primary_input_hashes.tsv`. Run from the repository root:

```bash
export CANDIDATES=/path/to/candidate_common_measurement_sample.tsv
export ALPS=/path/to/ALPS_primary_numeric_cohort_345.tsv
export WMH=/path/to/WMH_analysis_membership.tsv
export DTI=/path/to/jhu_subject_tract_FA_MD_AD_RD_with_demographics.tsv
export FS=/path/to/FreeSurfer_atrophy_analysis_table.tsv
export OUT=/path/to/new/private/run
bash workflow/run_primary.sh
```

Keep generated files outside the repository. The run produces participant-level as well as aggregate tables, so do not publish the run directory. The primary analysis evaluates the expanded and supported cohorts and reports the supported 307-participant sample.

After a primary run, the separate control-cohort check can be run with:

```bash
python analysis/diffusion/control_cohort_fa.py \
  --cohort "$OUT/tables/COMMON_subject_measurements_and_membership.tsv" \
  --tracts "$DTI" --out /path/to/new/private/control_cohort_FA_contrasts.tsv
```

This estimates an age- and sex-adjusted difference between the control cohorts; it does not estimate a scanner-only offset.

## Verification and release status

Run `python3 tests/check_integrity.py` to verify the recovered source hashes and syntax. These checks do not rerun the manuscript analyses. The original scripts are preserved where included; some follow-up scripts retain dependencies on the original project layout. See [source review](docs/review.md) and [release inventory](docs/release_inventory.md) for what remains to reconcile.

## Data and generated results

Study inputs and derived participant-level data are governed by the ADNI and FTLDNI/NIFD access agreements. The repository contains no study input tables, generated CSV/TSV result tables, images, or run logs. Generate results locally using the documented inputs and keep them outside the repository. The TSV files under `provenance/` record checksums and source information.

## Credits and citation

**Study authors:** Oumayma Soula, Nawress Khlifa, Ahmed Rebai, and Udunna C. Anazodo.

**Data sources:** Alzheimer’s Disease Neuroimaging Initiative (ADNI) and Frontotemporal Lobar Degeneration Neuroimaging Initiative (FTLDNI/NIFD). We thank the participants and the teams who collected and maintain these data.

Use [`CITATION.cff`](CITATION.cff) to cite the software. Cite the companion paper separately when a published reference is available.

## License

The code license has not yet been finalized. Confirm the license, upstream execution records, and the figure-to-output mapping before public release.
