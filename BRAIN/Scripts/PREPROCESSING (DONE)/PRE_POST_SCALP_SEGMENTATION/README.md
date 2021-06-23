Flap segmentation is done for pre/post-operative MRI files that already have scalp extracted from SINGLE_SET_ANALYSIS/Python(DRAFT)/single_set_scalp_and_brain.py.

The purpose is to make a list of clustered files, each predicting a flap region, so that practitioners can pick the most accurate file to use as ground truth.

See README.md at MNI_Y:N_scalp_comparison folder to check the justification for making MNI alignment of scalp files optional rather than required.

Major comments on the python script:

# Comment 1: Ranges for epsilon value and minimum points have to be optimized with GPU support because the code takes more than 2 hours on laptop to run once.

# Comment 2: Experimentally, epsilon value should be at least 6, and minimum points should be greater than 10. The optimal values cannot be calculated with laptop alone given comment 1 above. 

Once optimal values for epsilon and minimum points are identified, further preprocessing such as convexhull 3D (for generating skull flap-like mesh from cluster of points) can be performed. Whether convexhull-modified  or -unmodified data will be used as GT for training the prediction algorithm is to be determined after completing the script for algorithm. 
