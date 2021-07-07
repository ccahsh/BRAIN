

import os
import re
import vtk
import sys
import csv
import math
import shutil
import numpy as np
import pandas as pd
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


# defining nii to stl conversion

def nii_2_mesh(filename_nii, filename_stl, label):

    try:
        
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(filename_nii)
        reader.Update()
        
        surf = vtk.vtkDiscreteMarchingCubes()
        surf.SetInputConnection(reader.GetOutputPort())
        surf.SetValue(0, label)
        surf.Update()
        
        smoother= vtk.vtkWindowedSincPolyDataFilter()
        if vtk.VTK_MAJOR_VERSION <= 5:
            smoother.SetInput(surf.GetOutput())
        else:
            smoother.SetInputConnection(surf.GetOutputPort())
        smoother.SetNumberOfIterations(30)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.GenerateErrorScalarsOn()
        smoother.Update()
         
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(smoother.GetOutputPort())
        writer.SetFileTypeToASCII()
        writer.SetFileName(filename_stl)
        writer.Write()
        
    except:
        
        pass



if __name__ == "__main__":

# Before: HD-BET Installation and Identification

    if not os.path.exists("HD-BET"):

        os.system("git clone https://github.com/MIC-DKFZ/HD-BET")
        os.system("cd HD-BET")
        os.system("pip3 install -e .")
        os.system("cd ../")

    introduction = input("\nYou need to give cluster classification number, pre-operative and post-operative SCALP files. MNI152 raw file is optional. Press enter to continue.\n\n")

    cluster_label = input("\n(REQUIRED) Type in the number you would like to classify the segmented FLAP files with, from 000 to 999. Make sure the number is not repeated in other cluster folders. If repeated, the folder will be overwritten.\n\n")

    if not os.path.exists('Clusters_' + str(cluster_label)):
        os.mkdir('Clusters_' + str(cluster_label))
        
    preop_name = input("\n(REQUIRED) Type in the name of the pre-operative SCALP file. Make sure you include nii.gz format.\n\n")

    postop_name = input("\n(REQUIRED) Type in the name of the post-operative SCALP file. Make sure you include nii.gz format.\n\n")

    MNI_registration = input("\n(OPTIONAL) Type Y if you would like to register pre-operative and post-operative SCALP files to MNI152 SCALP file. If not, press enter instead. We do not recommend MNI normalization because the predicted flap region size may be insufficient.\n\n")

    # Optional: Task 0 : MNI Scalp Extraction and Input Normalization

    # Comment on Task 0 : From testing, Task 0 may not give significant effect on scalp segmentaation.

    if (MNI_registration == 'Y'):

    # Task 0.1 : MNI brain segmentation.

        MNI_original = input("\nType in the name of the unprocessed MNI152 file.\n\n")
        os.rename(MNI_original, "MNI-template.nii.gz")
        os.system("hd-bet -i MNI-template.nii.gz -device cpu -mode fast -tta 0")

    #remove "-device cpu -mode fast -tta 0" if GPU support is available.
    #install HD-BET from https://github.com/MIC-DKFZ/HD-BET. HD-BET is the most up-to-date brain segmentation algorithm (6/17/21).

    # Task 0.2 : Making MNI SCALP file.

        os.remove("MNI-template_bet.nii.gz")
        os.rename("MNI-template_bet_mask.nii.gz", "MNI-template_BRAINMASK.nii.gz")

        brain_mask = nib.load('MNI-template_BRAINMASK.nii.gz')
        MNI_ref = nib.load('MNI-template.nii.gz')

        brain_mask_A = np.array(brain_mask.dataobj)
        MNI_ref_A = np.array(MNI_ref.dataobj)

        # Task 0.2.1 : Dimension check.

        if(brain_mask_A.shape == MNI_ref_A.shape):
            
            for x in range(0, brain_mask_A.shape[0]-1):
                for y in range(0, brain_mask_A.shape[1]-1):
                    for z in range(0, brain_mask_A.shape[2]-1):
                        if(brain_mask_A[x][y][z] > 0):
                            MNI_ref_A[x][y][z] = 0
                    
        else:

            print("Comparison not possible due to difference in dimensions.")
            
        # Task 0.2.2 : Volume Restriction.
        
        for x in range(0, MNI_ref_A.shape[0]-1):
            for y in range(0, MNI_ref_A.shape[1]-1):
                for z in range(0, MNI_ref_A.shape[2]-1):
                    if(x < ((MNI_ref_A.shape[0]-1)*0.03) or x > ((MNI_ref_A.shape[0]-1)*0.96) or y < ((MNI_ref_A.shape[1]-1)*0.01) or y > ((MNI_ref_A.shape[1]-1)*0.99) or z < ((-(MNI_ref_A.shape[2]-1)*y*0.000275)+85)):
                            MNI_ref_A[x][y][z] = 0
                        
        # Task 0.2.3 : Maximum value check.
        
        def paraMAX():
            M = 0
            for x in range(int(0.05*(MNI_ref_A.shape[0]-1)),int(0.95*(MNI_ref_A.shape[0]-1))):
                for y in range(int(0.05*(MNI_ref_A.shape[1]-1)),int(0.95*(MNI_ref_A.shape[1]-1))):
                    for z in range(int(0.05*(MNI_ref_A.shape[2]-1)),int(0.95*(MNI_ref_A.shape[2]-1))):
                       if(M < MNI_ref_A[x][y][z]):
                            M = MNI_ref_A[x][y][z]
            return M
            
        # Task 0.2.4 : Filtering by maximum threshold.
        
        MAX = paraMAX()
        MAX_thres = 0.225*MAX
        
        for x in range(0, MNI_ref_A.shape[0]-1):
            for y in range(0, MNI_ref_A.shape[1]-1):
                for z in range(0, MNI_ref_A.shape[2]-1):
                    if(MNI_ref_A[x][y][z] < MAX_thres):
                        MNI_ref_A[x][y][z] = 0
        
        # Task 0.2.5 : Removing non-scalp voxels by area inspection.
        
        ns_thres = 0.34*MAX
        
        for x in range(1, MNI_ref_A.shape[0]-1):
            for y in range(1, MNI_ref_A.shape[1]-1):
                for z in range(1, MNI_ref_A.shape[2]-1):
                    M = 0
                    for k in range(-1,2):
                        for m in range(-1,2):
                            for n in range(-1,2):
                                if MNI_ref_A[x+k][y+m][z+n] >= M:
                                    M = MNI_ref_A[x+k][y+m][z+n]
                    if M < ns_thres:
                        MNI_ref_A[x][y][z] = 0
        
        # Task 0.2.6 : Extraction
        
        MNI_scalp_array = nib.Nifti1Image(MNI_ref_A, affine=np.eye(4))
        nib.save(MNI_scalp_array, "MNI-template_SCALP.nii.gz")

    # Task 0.3 : Aligning pre-operative and post-operative SCALP files onto MNI SCALP file.

        flt1 = fsl.FLIRT(bins=640, cost_func='mutualinfo')
        flt1.inputs.in_file = preop_name
        flt1.inputs.reference = 'MNI-template_SCALP.nii.gz'
        flt1.inputs.output_type = "NIFTI_GZ"
        flt1.cmdline
        res = flt1.run()

        os.remove(str(preop_name[:-7]) + '_flirt.mat')
        preop_name = str(preop_name[:-7]) + '_flirt.nii.gz'
        
        flt2 = fsl.FLIRT(bins=640, cost_func='mutualinfo')
        flt2.inputs.in_file = postop_name
        flt2.inputs.reference = 'MNI-template_SCALP.nii.gz'
        flt2.inputs.output_type = "NIFTI_GZ"
        flt2.cmdline
        res = flt2.run()

        os.remove(str(postop_name[:-7]) + '_flirt.mat')
        preop_name = str(postop_name[:-7]) + '_flirt.nii.gz'



    # Task 1 : Intensity Normalization, Difference NIFTI Generation, and Mesh

    preop = nib.load(preop_name) #might need to change to normalized name
    A = np.array(preop.dataobj)  #might need to change to normalized name

    postop = nib.load(postop_name)
    B = np.array(postop.dataobj)

    normal_diff = np.zeros((A.shape[0],A.shape[1],A.shape[2]))

    AA = A/(np.max(A))
    BB = B/(np.max(B))

    MIN_thres = 0.34

    # Aligning postoperative SCALP file to preoperative SCALP file lowers accuracy, so it is avoided in this script. Minimum threshold for absolute intensity difference is manually selected after testing with pre-operative and post-operative T1w files (Patient 3) in OPENNEURO Database: https://openneuro.org/datasets/ds001226/versions/1.0.0 and https://openneuro.org/datasets/ds002080/versions/1.0.1. This is subject to change, depending on whether T1w is registered with T2w before analysis or not. However, skull segmentation is better with T1w only (confirmed).

    for x in range(0,AA.shape[0]-1):
        for y in range(0,AA.shape[1]-1):
            for z in range(0,AA.shape[2]-1):
                if abs(AA[x][y][z]-BB[x][y][z] > MIN_thres): #andBB[x][y][z] == 0
                    normal_diff[x][y][z] = 1
                else:
                    normal_diff[x][y][z] = 0
                    
    normal_diff_nib = nib.Nifti1Image(normal_diff, affine=np.eye(4))
    nib.save(normal_diff_nib, "normalized_difference_MNIenabled.nii.gz")

    filename_nii = 'normalized_difference_MNIenabled.nii.gz'
    filename_stl = filename_nii[:-7] + '.stl'
    label = 1

    nii_2_mesh(filename_nii, filename_stl, label)
    shutil.move('normalized_difference_MNIenabled.stl','Clusters_' + str(cluster_label))

    # Completion 1 notified.

    confirmation1 = input("\nNormalized, unprocessed FLAP File generated and mesh generated (relocated). Press enter to continue.")



    # Task 2 : DBSCAN Segmentation (C++ implementation)

    print("\nFor DBSCAN segmentation, epsilon value is 8, and minpoints value is 40. For eps and minpts, edit main.cpp if needed. For size thresholds, edit this script.")

    a = nib.load('normalized_difference_MNIenabled.nii.gz')
    A = np.array(a.dataobj)

    valid_coord = []

    for x in range(0,A.shape[0]-1):
        for y in range(0,A.shape[1]-1):
            for z in range(0,A.shape[2]-1):
                if (A[x][y][z] == 1):
                    valid_coord += [[x,y,z]]

    np.savetxt("valid_coord.csv", valid_coord, delimiter=",")

    with open('valid_coord.csv', newline='') as csvfile:
        data = np.array(list(csv.reader(csvfile)))
        data_float = data.astype(float)

    add = np.array([[data_float.shape[0], ' ', ' ']])

    DBSCAN_prep = np.concatenate((add, data_float))

    np.savetxt("valid_coord.dat", DBSCAN_prep, fmt='%s', delimiter=',')

    os.remove("valid_coord.csv")

    df = pd.read_csv("valid_coord.dat", sep=";", header=0)

    df.rename(columns = lambda x: re.sub('\D','',x),inplace=True)

    df.to_csv("valid_coord_DBSCANprep.dat", sep = ' ', index = False)

    os.remove("valid_coord.dat")

    os.system("g++ main.cpp dbscan.cpp -o DBSCANcsv")

    os.system("./DBSCANcsv")

    os.remove("valid_coord_DBSCANprep.dat")

    # Cluster organization (cluster_label, Ashape, and valid_coord given from input)

    DBSCANraw = np.genfromtxt('DBSCAN_raw.csv')

    rownum = int(DBSCANraw.shape[0]/4 - 1)

    cluster_max = 0

    for i in range(rownum + 1):
        if DBSCANraw[4*i + 3] >= cluster_max:
            cluster_max = int(DBSCANraw[4*i + 3])
        
    cluster_lists = [[] for i in range(cluster_max + 1)]

    for i in range(rownum + 1):
        if DBSCANraw[4*i + 3] >= 1:
            cluster_lists[int(DBSCANraw[4*i + 3])].append([valid_coord[i]])
        
    for r in range(1,cluster_max + 1):
        cluster_indi = np.array(cluster_lists[r])
        cluster_coord = np.zeros((A.shape[0], A.shape[1], A.shape[2]))
        for s in range(len(cluster_indi)):
            cluster_coord[cluster_indi[s][0][0],cluster_indi[s][0][1],cluster_indi[s][0][2]] = 1
        if len(cluster_indi) >= 10000:
            cluster_nib = nib.Nifti1Image(cluster_coord, affine=np.eye(4))
            nib.save(cluster_nib, "DBSCAN-cluster" + str(r) + ".nii.gz")
            
            filename_nii = "DBSCAN-cluster" + str(r) + ".nii.gz"
            filename_stl = filename_nii[:-7] + '.stl'
            label = 1
            
            nii_2_mesh(filename_nii, filename_stl, label)
                
            shutil.move("DBSCAN-cluster" + str(r) + ".nii.gz", 'Clusters_' + str(cluster_label))
            shutil.move("DBSCAN-cluster" + str(r) + ".stl", 'Clusters_' + str(cluster_label))

    os.remove("DBSCAN_raw.csv")
    shutil.move('normalized_difference_MNIenabled.nii.gz','Clusters_' + str(cluster_label))






