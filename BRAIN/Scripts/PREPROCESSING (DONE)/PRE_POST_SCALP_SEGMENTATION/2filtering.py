import numpy as np
import nibabel as nib

def func(a):
    thres = 0.37
    A = nib.load(str(a))
    A = np.array(A.dataobj)
    B = np.copy(A)
    B = B/np.max(B)
    for x in range(B.shape[0]):
        for y in range(B.shape[1]):
            for z in range(B.shape[2]):
                if B[x][y][z] < thres:
                    B[x][y][z] = 0
    BB = nib.Nifti1Image(B, affine=np.eye(4))
    nib.save(BB, str(a)[:2] + "_thres_" + str(thres) + ".nii.gz")

func('13_T1w_post_SCALP_normalized_filtered_nonMNI_difference_post=0_reflection.nii.gz')
func('16_T1w_post_SCALP_normalized_filtered_nonMNI_difference_post=0_reflection.nii.gz')
func('26_T1w_post_SCALP_normalized_filtered_nonMNI_difference_post=0_reflection.nii.gz')

