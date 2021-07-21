import numpy as np
import nibabel as nib

def func(a):
    A = nib.load(str(a))
    A = np.array(A.dataobj)
    B = np.copy(A)
    B = B/np.max(B)
    C = np.zeros((B.shape[0],B.shape[1],B.shape[2]))
    center = [129.99408517, 155.40402737140525, 120.06418584]
    for x in range(B.shape[0]):
            for y in range(B.shape[1]):
                    for z in range(B.shape[2]):
                        if 10 <= x < B.shape[0]-10 and 10 <= round(2*center[0]-x) < B.shape[0]-10:
                            if B[round(2*center[0]-x)][y][z] != 0:
                                Val = []
                                for i in range(-10,11):
                                    if B[x+i][y][z] != 0:
                                          Val += 'F'
                                if not 'F' in Val:
                                    C[x][y][z] = 1
    CC = nib.Nifti1Image(C, affine=np.eye(4))
    nib.save(CC, str(a)[:2] + "_reflecting.nii.gz")

#26th patient smaple failed. Other two made it successfully.
#maybe fill in the space between reflected file and scalp diff file using rays from origin. that'll ensure dbscan success.
