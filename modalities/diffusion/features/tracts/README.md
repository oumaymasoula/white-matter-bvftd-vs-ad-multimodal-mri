# TBSS and JHU tract features

These recovered scripts retain the original 347-volume cohort assumptions.
Use the original subject/volume order and acquisition-support manifests.
Set WM_TBSS_ROOT to the work directory. Supply scheduler account and log paths
at submission. For the extraction launcher, set WM_TRACT_EXTRACTOR to the
absolute path of the repository extractor and configure WM_COARSE_4MM_SUBJECT
and WM_COARSE_5MM_SUBJECT privately using the original acquisition records.
The original task, dataset and group checks for these sensitivity flags
are retained.

The stages expect existing FA and scalar inputs, registration and projection
manifests, acquisition support masks and the negative-value audit.
Some of those preparation steps and all participant-level inputs are outside
this package. See the script environment variables before running the
coverage calculator or extractor. The extractor writes participant-level
tract features; direct its outputs to private storage.
