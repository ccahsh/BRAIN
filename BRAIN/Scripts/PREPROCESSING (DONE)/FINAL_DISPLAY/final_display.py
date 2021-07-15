#1. Extract scalp from preop (raw) file. Do HD-BET, normalization, filtering without MNI registration. Let the output file be 'preop-scalp (raw)'.
#2. Perform FLIRT from 'preop-scalp (raw)' to MNI scalp template (normalized and filtered) so as to remove non-skull regions (e.g. neck). Let the output file be 'MNI-registered preop-scalp (raw)'. Save the 4x4 affine transform matrix.
#3. Make the inverse matrix of the 4x4 affine transform matrix (mentioned in the previous step) using numpy.linalg.inv function.
#4. Perform FLIRT-applyxfm (what is applyxfm: using a known affine transform matrix) (use inverse matrix found above) to 'MNI-registered preop-scalp (raw)' with reference to 'preop-scalp (raw)' (why: to keep the same size). Let the output file be 'preop-scalp (return)'.
#5. Perform FLIRT from MNI scalp template (normalized and filtered) to 'preop-scalp (return)'. With non-skull regions removed, the transformation is now suitable for the SEGMENTED SCALP, too. Let the 4x4 affine transform matrix output (-omat) be 'MNI-return.mat'.
#6. Perform FLIRT-applyxfm (use 'MNI-return.mat' that is mentioned above) to SEGMENTED SCALP with reference to 'preop-scalp (raw)'. Let the output file be 'SEGMENTED SCALP-raw'.
#7. Overlap 'SEGMENTED SCALP-raw' and 'preop-scalp (raw)' in MRIcroGL to check if the flap sits on the scalp.

import os
import numpy as np
import nibabel as nib


def segmentation():

    T1w_name = input('\nType in the name of unmodified (raw) T1w file.\n')
    flap_name = input('\nType in the name of file containing the segmented, MNI-registered flap.\n')
    MNI_scalp_name = input('\nType in the name of referenced MNI scalp file. We recommend to have the scalp extracted from the functions provided in the repository (will be organized).\n')
    GPU_enabled = input('\nIs GPU enabled? Type Y/N.\n')
    
    # Task 1: Brain segmentation and deletion
    
    if (GPU_enabled == 'Y'):
        os.system("hd-bet -i " + str(T1w_name))
    else:
        if (GPU_enabled == 'N'):
            os.system("hd-bet -i " + str(T1w_name) + " -device cpu -mode fast -tta 0")
    
    os.remove(T1w_name[:-7] + "_bet.nii.gz")
    os.rename(T1w_name[:-7] + "_bet_mask.nii.gz", T1w_name[:-7] + "_BRAINMASK.nii.gz")

    brain_mask = nib.load(T1w_name[:-7] + "_BRAINMASK.nii.gz")
    t1w_t2w = nib.load(T1w_name)

    brain_mask_A = np.array(brain_mask.dataobj)
    t1w_t2w_A = np.array(t1w_t2w.dataobj)

    # 1.1 : Checking dimensional congruency between brain mask and overlaid file.

    if(brain_mask_A.shape == t1w_t2w_A.shape):

    # 1.2 : Removing brain from overlaid file.
        
        for x in range(0, brain_mask_A.shape[0]-1):
            for y in range(0, brain_mask_A.shape[1]-1):
                for z in range(0, brain_mask_A.shape[2]-1):
                    if(brain_mask_A[x][y][z] > 0):
                        t1w_t2w_A[x][y][z] = 0
                
    else:

        print("Comparison not possible due to difference in dimensions.")
        
    # 1.3 : Isolating scalp with enclosed coordinate volume.

    for x in range(0, t1w_t2w_A.shape[0]-1):
        for y in range(0, t1w_t2w_A.shape[1]-1):
            for z in range(0, t1w_t2w_A.shape[2]-1):
                if(x < ((t1w_t2w_A.shape[0]-1)*0.03) or x > ((t1w_t2w_A.shape[0]-1)*0.96) or y < ((t1w_t2w_A.shape[1]-1)*0.01) or y > ((t1w_t2w_A.shape[1]-1)*0.99) or z < ((-(t1w_t2w_A.shape[2]-1)*y*0.000275)+85)):
                    t1w_t2w_A[x][y][z] = 0
                    
    # 1.4 : Finding value of threshold intensity for scalp segmentation.

    def paraMAX():
        M = 0
        for x in range(int(0.05*(t1w_t2w_A.shape[0]-1)),int(0.95*(t1w_t2w_A.shape[0]-1))):
            for y in range(int(0.05*(t1w_t2w_A.shape[1]-1)),int(0.95*(t1w_t2w_A.shape[1]-1))):
                for z in range(int(0.05*(t1w_t2w_A.shape[2]-1)),int(0.95*(t1w_t2w_A.shape[2]-1))):
                   if(M < t1w_t2w_A[x][y][z]):
                        M = t1w_t2w_A[x][y][z]
        return M
        
    MAX = paraMAX()
    MAX_thres = MAX*0.225

    # 1.5 : Segmenting scalp using threshold intensity.

    for x in range(0, t1w_t2w_A.shape[0]-1):
        for y in range(0, t1w_t2w_A.shape[1]-1):
            for z in range(0, t1w_t2w_A.shape[2]-1):
                if(t1w_t2w_A[x][y][z] < MAX_thres):
                    t1w_t2w_A[x][y][z] = 0
                    
    # Task 1.6 : Removing non-scalp voxels by area inspection.

    ns_thres = MAX*0.34

    for x in range(1, t1w_t2w_A.shape[0]-1):
        for y in range(1, t1w_t2w_A.shape[1]-1):
            for z in range(1, t1w_t2w_A.shape[2]-1):
                M = 0
                for k in range(-1,2):
                    for m in range(-1,2):
                        for n in range(-1,2):
                            if t1w_t2w_A[x+k][y+m][z+n] >= M:
                                M = t1w_t2w_A[x+k][y+m][z+n]
                if M < ns_thres:
                    t1w_t2w_A[x][y][z] = 0
     
    # Task 1.7 : Extraction

    scalp_array = nib.Nifti1Image(t1w_t2w_A, affine=np.eye(4))
    nib.save(scalp_array, T1w_name[:-7] + "_SCALP.nii.gz")
    os.remove(T1w_name[:-7] + "_BRAINMASK.nii.gz")
    
    # Task 2 : Getting inverse of (prscalp_raw --> MNI_scalp) affine transform matrix.
    
    prescalp_raw_name = T1w_name[:-7] + "_SCALP.nii.gz"
        
    MNI_registered_prescalp_raw_name = "MNI_registered_" + prescalp_raw_name
    
    os.system("flirt -in " + str(prescalp_raw_name) + " -ref " + str(MNI_scalp_name) + " -out " + str(MNI_registered_prescalp_raw_name) + " -omat prescalp_raw_to_MNI.mat -bins 640 -searchcost mutualinfo")
        
    inv = np.linalg.inv(np.loadtxt('prescalp_raw_to_MNI.mat'))
    
    np.savetxt('prescalp_raw_to_MNI_inv.mat',inv)
    
    prescalp_return_name = T1w_name[:-7] + "_return_SCALP.nii.gz"
    
    os.system("flirt -in " + str(MNI_registered_prescalp_raw_name) + " -ref " + str(prescalp_raw_name) + " -out " + str(prescalp_return_name) + " -init prescalp_raw_to_MNI_inv.mat -applyxfm")
    
    os.system("flirt -in " + str(MNI_scalp_name) + " -ref " + str(prescalp_return_name) + " -out delete.nii.gz -omat MNI-return.mat -bins 640 -searchcost mutualinfo")

    os.remove('delete.nii.gz')
    
    segmented_flap_raw_name = flap_name[:-7] + "_raw.nii.gz"
    
    os.system("flirt -in " + str(flap_name) + " -ref " + str(prescalp_raw_name) + " -out " + str(segmented_flap_raw_name) + " -init MNI-return.mat -applyxfm")
    # reference could be 'prescalp_return_name', but would not make visible difference.
    
    print("Completed. The name of the output flap file is : " + str(segmented_flap_raw_name) + ". Check if this file and unmodified (raw) T1w file overlap appropriately in external softwares (e.g. MRIcroGL) or python libraries.")
        
segmentation()
