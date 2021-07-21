import os
import vtk
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn import metrics
import plotly.express as px
import matplotlib.pyplot as plt
from nipype.interfaces import fsl
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from nipype.testing import example_data
from mpl_toolkits.mplot3d import Axes3D


def reflection_filter_merge(a):
    
    #task 1
    post_scalp_name = str(a)
    post_scalp = nib.load(post_scalp_name)
    post_scalp = np.array(post_scalp.dataobj)
    post_scalp = post_scalp/np.max(post_scalp)

    normal_diff = np.zeros((post_scalp.shape[0],post_scalp.shape[1],post_scalp.shape[2]))
    
    leftover = np.zeros((post_scalp.shape[0],post_scalp.shape[1],post_scalp.shape[2]))
        
    center = [129.99408517, 155.40402737140525, 120.06418584]
    
    for x in range(normal_diff.shape[0]):
        for y in range(normal_diff.shape[1]):
            for z in range(normal_diff.shape[2]):
                if (z < (-(normal_diff.shape[2]-1)*y*0.000275) + 189):
                    normal_diff[x][y][z] = post_scalp[x][y][z]
                else:
                    leftover[x][y][z] = post_scalp[x][y][z]

    #task 2
    thres = 0.37
    B = np.copy(normal_diff)
    for x in range(B.shape[0]):
        for y in range(B.shape[1]):
            for z in range(B.shape[2]):
                if B[x][y][z] < np.max(B)*thres:
                    B[x][y][z] = 0
    
    reconstructed = np.zeros((post_scalp.shape[0],post_scalp.shape[1],post_scalp.shape[2]))
    
    for x in range(reconstructed.shape[0]):
        for y in range(reconstructed.shape[1]):
            for z in range(reconstructed.shape[2]):
                if B[x][y][z] != 0:
                    reconstructed[x][y][z] = B[x][y][z]
                if leftover[x][y][z] != 0:
                    reconstructed[x][y][z] = leftover[x][y][z]
    
    Reconstructed_nib = nib.Nifti1Image(reconstructed, affine=np.eye(4))
    nib.save(Reconstructed_nib, str(a)[:2] + "_thres_" + str(thres) + ".nii.gz")
    
    # Completion 5 notified. MNI-normalized SCALP EXTRACTION COMPLETE.

reflection_filter_merge('13_T1w_post_SCALP_normalized_filtered.nii.gz')
reflection_filter_merge('16_T1w_post_SCALP_normalized_filtered.nii.gz')
reflection_filter_merge('26_T1w_post_SCALP_normalized_filtered.nii.gz')


 
