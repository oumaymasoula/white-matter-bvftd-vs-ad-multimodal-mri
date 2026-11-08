# Workflow scope and configuration

The repository separates image processing by modality and places statistical
models, cohort assembly and plotting in `analysis/`. The primary analysis
entry point starts from authorized feature tables.

| Branch | Included processing | Configuration |
|---|---|---|
| Diffusion | DWI preparation and final EDDY subject scripts | Set `WM_PROJECT_ROOT` and the relevant `ADNI_BIDS_ROOT` or `NIFD_BIDS_ROOT` for DWI preparation. |
| T1 | FreeSurfer subject reconstruction and feature extraction | For reconstruction set `WM_PROJECT_ROOT`, the relevant BIDS root and `FS_LICENSE`. |
| FLAIR | MARS-WMH segmentation using paired FLAIR and T1 | Set `WM_SCRATCH_ROOT` to the root that contains the expected BIDS, project and container directories. |

The subject runners take dataset, group and subject as positional arguments.
They retain the original cluster module requirements and expect the same
directory and intermediate-file conventions as the source scripts. Paths
were parameterized; the adapted runners have not been rerun on imaging data.
Both the archived and adapted file hashes are recorded in
`provenance/preprocessing_sources.tsv`.

This is a partial upstream release. Synb0 launcher selection and the ADNI ALPS
link to the primary model inputs remain unverified. TBSS QC records support a
0.20 mean-FA skeleton threshold, a 25% JHU atlas threshold, at least 100
skeleton voxels per tract, and at least 95% weighted and unweighted acquisition
support. The 99% rule is a sensitivity check. The recovered TBSS and JHU stages are included, with prerequisites documented in
[Diffusion feature extraction](../modalities/diffusion/features/README.md). Manuscript figure exports and raw audit files stay separate.
