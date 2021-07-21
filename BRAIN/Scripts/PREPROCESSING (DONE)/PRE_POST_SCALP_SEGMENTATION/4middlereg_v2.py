from __future__ import division
from sympy import *
import numpy as np
import nibabel as nib

def middle_reg(a,b):
  
    A = nib.load(str(a))
    AA = np.array(A.dataobj)
    B = []
    for x in range(AA.shape[0]):
        for y in range(AA.shape[1]):
            for z in range(AA.shape[2]):
                if AA[x][y][z] != 0:
                    B += [[x,y,z]]

    C = nib.load(str(b))
    CC = np.array(C.dataobj)
    D = []
    center = [129.99408517, 155.40402737140525, 120.06418584]
    #make sure the center for the actual code is the variable stored in the full code version.
    for x in range(CC.shape[0]):
        for y in range(CC.shape[1]):
            for z in range(CC.shape[2]):
                if CC[x][y][z] != 0:
                    D += [[x,y,z]]
    E = []
      #final collection of MNI-mid registrated points
    for i in range(len(B)):
        F = []
        K = []
        for l in range(150,250):
        #range is manually determined
            Bx = round(center[0] + (B[i][0]-center[0])*(l/200))
            By = round(center[1] + (B[i][1]-center[1])*(l/200))
            Bz = round(center[2] + (B[i][2]-center[2])*(l/200))
            K += [[Bx,By,Bz]]
        for m in range(len(D)):
            if D[m] in K:
                sum_abs = abs(D[m][0]-center[0]) + abs(D[m][1]-center[1]) + abs(D[m][2]-center[2])
                F += [[sum_abs,D[m]]]
        if len(F) != 0:
            F.sort()
            final_point = [round((F[0][1][0]+F[-1][1][0])/2), round((F[0][1][1]+F[-1][1][1])/2), round((F[0][1][2]+F[-1][1][2])/2)]
            E += [final_point]
    G = np.zeros((AA.shape[0],AA.shape[1],AA.shape[2]))
    for h in range(len(E)):
        G[E[h][0]][E[h][1]][E[h][2]] = 1
    J = nib.Nifti1Image(G,affine=np.eye(4))
    nib.save(J,str(a)[:-7]+"_middle_registered.nii.gz")


if __name__ == "__main__":
    middle_reg('13_reflecting.nii.gz','MNI-template_SCALP_normalized_filtered.nii.gz')
    middle_reg('16_reflecting.nii.gz','MNI-template_SCALP_normalized_filtered.nii.gz')
    middle_reg('26_reflecting.nii.gz','MNI-template_SCALP_normalized_filtered.nii.gz')

