# Inputs and outputs

The primary entry point accepts tab-separated tables. Keys are `dataset` and `subject`; diagnosis labels are checked against the candidate table. The tract table adds `tract` to its key. The candidate table represents the 337 records with complete required measurements; the analysis derives the 307-record site-supported subset.

| Variable | Private input | Content used |
|---|---|---|
| CANDIDATES | candidate_common_measurement_sample.tsv | Cohort, participant and diagnosis keys |
| ALPS | ALPS_primary_numeric_cohort_345.tsv | Bilateral ALPS and diffusion protocol |
| WMH | WMH_analysis_membership.tsv | WMH volume in mL, age, sex, site and scanner model |
| DTI | jhu_subject_tract_FA_MD_AD_RD_with_demographics.tsv | 20 JHU tracts, eligibility and FA/MD/RD/AD means |
| FS | FreeSurfer_atrophy_analysis_table.tsv | 14 structural measures and estimated intracranial volume |

Required column names and accepted aliases are defined in `analysis/multimodal/run_common_multimodal_models.py`. The five original input fingerprints are recorded in `provenance/primary_input_hashes.tsv`.

The workflow writes its results to `OUT/tables/`. Use `COMMON_group_counts.tsv` to check cohort counts, `COMMON_model_diagnostics.tsv` to check estimability, `COMMON_within_dataset_contrasts.tsv` for patient–control estimates, and `COMMON_cross_disease_contrasts.tsv` for differences between those effects. The membership table is a private analysis artifact and is the input to the control-cohort FA check.

Source images, subject lists, clinical tables, study outputs and processing logs are kept outside the repository.
