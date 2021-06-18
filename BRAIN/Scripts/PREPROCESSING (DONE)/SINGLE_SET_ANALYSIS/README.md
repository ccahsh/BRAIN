ALL INPUT MRI FILES MUST BE IN THE SAME DIRECTORY AS SCALP_BRAIN_SEGMENTATION.py without additional subfolders.

SCALP_BRAIN_SEGMENTATION.py segments scalp and brain mask from three inputs:

	1. T1w file (REQUIRED)
	2. T2w file (OPTIONAL)
	3. MNI152 reference file (REQUIRED)

T2w is set optional, so the python script can work both ways.

Tumor segmentation requires T1w, T2w, T1ce, and FLAIR files. T2w could be used only for tumor segmentation.
