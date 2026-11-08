# Source reconciliation findings

## Confirmed

- The supplied collection contains 281 files with no collection errors. No file was reported changed relative to the inventory.
- Both NIBI FA scripts match the executed FA follow-up provenance hashes. This recovers the corrected severity version accepting `YES`, rather than the unpatched local copy.
- The previously uploaded `clinical_imaging_307.tsv` hash matches the follow-up record: `43db7f6368c120b51322a8141d3e6e81d2da435fcf3cf7d7b14db9c563a5908b`. It is not included in this repository.
- The ADNI age-68 FA models were blocked by rank deficiency: rank 14, 15 parameters, 88 observations. NIFD models were evaluable with 45 observations, rank/parameters 8/8. Thus the cross-cohort contrasts were unavailable, not null. The redundant column has not yet been diagnosed; do not simply remove adjustment terms.
- The recovered primary code retains the ADNI three-group structure and separate cohort-specific adjustment. The recorded numerical environment is now available.

## Recovered implementation details, execution linkage pending

- `extract_jhu_tract_metrics.py` reads TBSS skeletonised FA/MD/AD/RD. It uses the JHU probabilistic atlas threshold 25%, intersected with the nonzero skeleton mask. It calculates unweighted arithmetic means and additional descriptive metrics. The script states a skeleton threshold of 0.20; verify the producer of the skeleton mask.
- Finite negative diffusivity values are preserved; eligibility and negative-fraction flags are recorded. Confirm the final negative-value checks and any reported sensitivity results. Do not describe values as clamped or removed.
- `run_eddy_final_subject.sh` loads FSL 6.0.7.18, uses eddy-rotated b-vectors, `04_preeddy.bval` and `dtifit --save_tensor`. Axial diffusivity is L1; radial diffusivity is (L2 + L3)/2. The actual selected b-values still require gradient files.
- `run_wmh_subject.sh` invokes a local `mars-wmh-nnunet.sif` with T1 and FLAIR through Apptainer. The checkpoint/container identity and final execution linkage remain unverified. This is evidence of an implementation, not complete segmentation validation.
- The recovered FreeSurfer runner loads 8.2.0-1. Confirm final subject logs, including retries, before asserting that every included scan used that version.
- `run_tensor_mni_alps.sh` points to `dwi_pilot`, `t1_pilot` and `alps_pilot`, uses FSL 6.0.7.7 and ANTs 2.6.5. It is not promoted as the final ALPS pipeline. Later tensor/registration scripts must be reconciled with final ALPS feature inputs.

## Excluded from the release

Pilot, single-subject test, before-fix, duplicate and unresolved scripts remain in the private source collection. The private source inventory records the original code-file decisions. Exclusion from this draft does not mean a script is incorrect. The first failed clinical association model and the pairwise anatomical severity variant are omitted; the site-supported and original-group-structure versions are included.

Accepted preprocessing repairs must be traced to final subject outputs. Their code may belong in the final repository even if generic production code alone looks cleaner. Do not erase their scientific provenance.

## Release blockers

1. Verify selected scan/diagnosis construction and all five exact primary inputs against recorded hashes. The candidate sample and corrected FreeSurfer preflight producers are recovered, but upstream diagnosis/scan selection is incomplete.
2. Recover final preprocessing launch records, accepted repair mappings, image QC decisions, gradient summaries and final ALPS implementation/ROI resources.
3. Resolve the five empty WMH masks. If genuine zero-burden cases were excluded, reconsider membership and rerun affected analyses.
4. Recover the 68-parcel regional producer, global-thickness/common-age frontal calibration producer, all main figure sources and final figure-to-table mappings. Recovered older anatomy scripts are candidates, not automatically the producer of every current result.
5. Rerun selected code on authorized inputs and compare coefficients, correction families, diagnostics and figure values. Current integrity checks do not substitute for this.
6. Refactor original absolute paths only after reference outputs are captured; validate the refactor against the originals. Add the complete software/container locks, an approved license, citation metadata and data/code statements.

## Next data collection

No second broad directory scan is needed. Use the existing inventory to select small provenance/QC/feature-construction records and the exact final aggregate output tables. The five primary inputs and subject-level tract table can be used privately for reproduction but must not be published automatically. Container files and imaging are not needed merely to inspect the methods.
