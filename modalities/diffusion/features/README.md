# Diffusion feature extraction

The final EDDY subject script fits the diffusion tensor and provides the FA,
MD, AD and RD maps. After tensor fitting, the tract and DTI-ALPS branches use
different spatial processing and region definitions.

- **Tracts:** TBSS preparation, nonlinear registration, post-registration
  skeletonization, prestats (FA threshold 0.20), non-FA projection, and JHU
  tract extraction. See [tract configuration](tracts/README.md).
- **DTI-ALPS:** The [reference script](dti_alps/reference/run_tensor_mni_alps.sh)
  shows tensor transformation through DWI-to-T1 and T1-to-MNI registration,
  then samples directional diffusivities in bilateral SCR and SLF ROIs to
  calculate left, right and mean ALPS. It uses the archived pilot directory
  conventions and requires the tensor, both registration stages, and the
  original ROI masks. Set WM_PROJECT_ROOT and ALPS_REPO_ROOT. This is a method
  reference; its link to the ALPS table used by the primary models has not
  been established.

The adapted scripts have not been rerun on imaging data. The downstream
statistical models are under the repository's analysis directory.
