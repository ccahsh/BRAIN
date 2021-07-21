UPDATE: Reflection (1,2,3,4 py files) should be used to post-operative scalp ONLY, and MUST be overlaid to scalp diff file RIGHT BEFORE DBSCAN ANALYSIS.

원래 scalp diff 에 뭘 추가: 
	1. ROI-specific scalp segmentation  (temporal region 집중: MNI registration 한
 	후 volume restriction --> higher threshold filtering --> muscle deletion)
	2. 전체 scalp (MNI registered, temporal 은 ROI-specific segmentation 후) 좌우 반전


Flap segmentation is done for pre/post-operative MRI files that already have scalp extracted from SINGLE_SET_ANALYSIS/Python(DRAFT)/single_set_scalp_and_brain.py.

HD-BET folder has been attached here to simplify command lines necessary for brain segmentation. 

The purpose is to make a list of clustered files, each predicting a flap region, so that practitioners can pick the most accurate file to use as ground truth.

See README.md at MNI_Y:N_scalp_comparison folder to check the justification for making MNI alignment of scalp files optional rather than required.

Once optimal values for epsilon and minimum points are identified, further preprocessing such as K-means and convexhull 3D (for generating skull flap-like mesh from cluster of points) can be performed. Whether convexhull-modified  or -unmodified data will be used as GT for training the prediction algorithm is to be determined after completing the script for algorithm. 

Note: scalp files should not be whitened to ensure minimal noise when predicting flap by taking difference between pre/postoperative scalp files.
