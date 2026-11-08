# Release inventory

This is a code-only review candidate. The source archive was preserved separately.

- Included: recovered modeling, matching, feature and plotting code; primary workflow; input contracts; numerical environment records; integrity checks.
- Removed from the candidate: rendered figures and the private source inventory.
- Withheld from the candidate: the recovered tract extractor and its Slurm launcher. The extractor contains participant-specific eligibility logic, and the launcher contains personal cluster paths. Recover the exact final version, move private identifiers into a private configuration, and verify identical feature outputs before including a portable extractor.
- Still to reconcile: final diffusion preprocessing, tensor reorientation and ALPS ROI workflow; FLAIR/WMH runner and model identity; accepted FreeSurfer repairs and launch configuration; final figure scripts; later manuscript sensitivity and QC scripts.
- Before release: match source files to the actual final runs, verify the code-only staging tree, and agree on the license and repository visibility.
