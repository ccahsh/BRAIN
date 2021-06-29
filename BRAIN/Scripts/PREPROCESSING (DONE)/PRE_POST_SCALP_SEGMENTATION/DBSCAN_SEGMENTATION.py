
# def functions are not used purposefully. Code will be condensed once it is approved. The code is verbose considering jit, but it is not critical.
#from line 326 - Comment 1: Ranges for epsilon value and minimum points have to be optimized with GPU support because the code takes more than 2 hours on laptop.


import os
import vtk
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn import metrics
from numba import jit, cuda
import plotly.express as px
import matplotlib.pyplot as plt
from nipype.interfaces import fsl
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from nipype.testing import example_data
from mpl_toolkits.mplot3d import Axes3D


@jit(target ="cuda")
def DBSCAN_Segmentation():

    # Before: HD-BET Installation and Identification

    if not os.path.exists("HD-BET"):

        os.system("git clone https://github.com/MIC-DKFZ/HD-BET")
        os.system("cd HD-BET")
        os.system("pip3 install -e .")
        os.system("cd ..")

    introduction = input("\nYou need to give cluster classification number, pre-operative and post-operative SCALP files. MNI152 raw file is optional. Press enter to continue.\n\n")

    cluster_label = input("\n(REQUIRED) Type in the number you would like to classify the segmented FLAP files with, from 000 to 999. Make sure the number is not repeated in other cluster folders.\n\n")

    preop_name = input("\n(REQUIRED) Type in the name of the pre-operative SCALP file. Make sure you include nii.gz format.\n\n")

    postop_name = input("\n(REQUIRED) Type in the name of the post-operative SCALP file. Make sure you include nii.gz format.\n\n")

    MNI_registration = input("\n(OPTIONAL) Type Y if you would like to register pre-operative and post-operative SCALP files to MNI152 SCALP file. If not, press enter instead. We recommend MNI normalization so as to make sure the GT models are congruent to the MRI template scalp file, which may be used as input for training algorithm.\n\n")

    # Optional: Task 0 : MNI Scalp Extraction and Input Normalization

    # Comment on Task 0 : From testing, Task 0 may not give significant effect on scalp segmentaation.

    if (MNI_registration == Y):

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
        MAX_thres = MAX*0.225
        
        for x in range(0, MNI_ref_A.shape[0]-1):
            for y in range(0, MNI_ref_A.shape[1]-1):
                for z in range(0, MNI_ref_A.shape[2]-1):
                    if(MNI_ref_A[x][y][z] < MAX_thres):
                        MNI_ref_A[x][y][z] = 0
        
        # Task 0.2.5 : Removing non-scalp voxels by area inspection.
        
        ns_thres = 0.34
        
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



    # Task 1 : Intensity Normalization and Difference NIFTI Generation

    preop = nib.load(preop_name) #might need to change to normalized name
    A = np.array(preop.dataobj)  #might need to change to normalized name

    postop = nib.load(postop_name)
    B = np.array(postop.dataobj)

    normal_diff = np.zeros((A.shape[0],A.shape[1],A.shape[2]))

    print(np.max(A))
    print(np.max(B))

    AA = A/(np.max(A))
    BB = B/(np.max(B))

    MIN_thres = 0.34

    # Aligning postoperative SCALP file to preoperative SCALP file lowers accuracy, so it is avoided in this script. Minimum threshold for absolute intensity difference is manually selected after testing with pre-operative and post-operative T1w files (Patient 3) in OPENNEURO Database: https://openneuro.org/datasets/ds001226/versions/1.0.0 and https://openneuro.org/datasets/ds002080/versions/1.0.1. This is subject to change, depending on whether T1w is registered with T2w before analysis or not.

    for x in range(0,AA.shape[0]-1):
        for y in range(0,AA.shape[1]-1):
            for z in range(0,AA.shape[2]-1):
                if abs(AA[x][y][z]-BB[x][y][z] > MIN_thres): #andBB[x][y][z] == 0
                    normal_diff[x][y][z] = 1
                else:
                    normal_diff[x][y][z] = 0
                    
    normal_diff_nib = nib.Nifti1Image(normal_diff, affine=np.eye(4))
    nib.save(normal_diff_nib, "normalized_difference_MNIenabled.nii.gz")

    # Completion 1 notified.

    confirmation1 = input("\nNormalized, unprocessed FLAP File generated and ready for mesh. Press enter to continue.")



    # Task 2 : NIFTI to Mesh Conversion

    # Task 2.1 : NIFTI to Mesh

    def nii_2_mesh(filename_nii, filename_stl, label):

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

    filename_nii = 'normalized_difference_MNIenabled.nii.gz'
        
    filename_stl = filename_nii[:-7] + '.stl'
        
    label = 1
        
    nii_2_mesh(filename_nii, filename_stl, label)
        
    # Task 2.2 : Mesh Relocation

    os.mkdir('Mesh_Visualization_' + str(cluster_label))
    shutil.move('normalized_difference_MNIenabled.stl','Mesh_Visualization_' + str(cluster_label))

    # Completion 2 notified.

    confirmation2 = input("Normalized, unprocessed FLAP mesh file generated and relocated. Ready for DBSCAN. Press enter to continue.")



    # Task 3 : DBSCAN Segmentation

    a = nib.load('rescaled_difference_MNIenabled.nii.gz')
    A = np.array(a.dataobj)

    # Task 3.1 : Listing coordinates of rescaled_difference NIFTI file into arrays

    valid_coord = []

    for x in range(0,A.shape[0]-1):
        for y in range(0,A.shape[1]-1):
            for z in range(0,A.shape[2]-1):
                if (A[x][y][z] == 1):
                    valid_coord += [[x,y,z]]

    #print(valid_coord)


    # Task 3.2 : Rescaling coordinates into [0,1]^3 volume.

    X = [v[0] for v in valid_coord]
    Y = [v[1] for v in valid_coord]
    Z = [v[2] for v in valid_coord]

    # Extra: plotting coordinate information into graph for visualization

    #fig, axs = plt.subplots(1, 3, figsize=(16,4), dpi=300)

    #axs[0].hist(X, bins=50, color='black', rwidth=0.9)
    #axs[0].set_title('X Coordinates')
    #axs[1].hist(Y, bins=50, color='black', rwidth=0.9)
    #axs[1].set_title('Y Coordinates')
    #axs[2].hist(Z, bins=50, color='black', rwidth=0.9)
    #axs[2].set_title('Z Coordinates')

    X_rescaled = []
    m = np.max(X)
    for i in range(len(X)):
        X_rescaled.append(X[i]/m)

    Y_rescaled = []
    n = np.max(Y)
    for j in range(len(Y)):
        Y_rescaled.append(Y[j]/n)
        
    Z_rescaled = []
    p = np.max(Z)
    for k in range(len(Z)):
        Z_rescaled.append(Z[k]/p)

    #print(X_rescaled)
    #print(Y_rescaled)
    #print(Z_rescaled)

    valid_coord_rescaled = []
    for e in range(len(X_rescaled)):
        valid_coord_rescaled.append([X_rescaled[e],Y_rescaled[e],Z_rescaled[e]])
      
    valid_coord_rescaled = np.array(valid_coord_rescaled)

    #print(valid_coord_rescaled)


    # Task 3.3 : Executing DBSCAN
    # Note: HDBSCAN may be implemented if necessary.

    # Task 3.3.1 : Setting up silhouette scores and combinations of epsilon / minimum samples

    S = []
    comb = []

    eps_r = range(5,75)
    minpts_r = range(3,200)

    # Comment 1: Ranges for epsilon value and minimum points have to be optimized with GPU support because the code takes more than 2 hours on laptop to run once.
    # Comment 2: Epsilon value should be at least 6, and minimum points should be greater than 10. The optimal values cannot be calculated given comment 1 above.

    for k in eps_r:
        for j in minpts_r:
            model = DBSCAN(eps = int(k)*0.01, min_samples = int(j))
            run = model.fit(valid_coord_rescaled)
            S.append(metrics.silhouette_score(valid_coord_rescaled, run.labels_, metric='euclidean'))
            comb.append([str(k),str(j)])
            

    # Task 3.3.2 : Finding maximum silhouette scores and corresponding [epsilon / minimum samples] values.

    Smax = 0
    Smax_set = []

    for q in range(len(eps_r)*len(minpts_r)):
        if (S[q] > Smax):
            Smax = S[q]
            Smax_set = comb[q]
            
    # Assume silhouette coefficient will be 1 mode for max. If not, the code is still valid because the epsilon value is smallest.

    k_max = Smax_set[0]
    #print(k_max)
    j_max = Smax_set[1]
    #print(j_max)


    # Task 3.3.3 : Performing DBSCAN

    maxmodel = DBSCAN(eps = int(k_max)*0.01, min_samples = int(j_max), metric = 'euclidean', metric_params = None, algorithm='auto', leaf_size = 30, p = None, n_jobs = None)

    # All parameters included for potential adjustments.

    scanned_results = maxmodel.fit(valid_coord_rescaled)
    confirmation33 = input("DBSCAN complete. Press enter to continue.")


    # Task 3.4 : Creating and storing cluster-based files for GT selection.

    os.mkdir('Clusters_' + str(cluster_label))

    cluster_max = max(scanned_results.labels_)

    cluster_lists = [[] for i in range(cluster_max+1)]

    for l in range(len(scanned_results)):
        if (scanned_results.labels_[l] >= 0):
            cluster_lists[scanned_results.labels_[l]].append(valid_coord[l])

    for r in range(cluster_max+1):
        cluster_indi = np.array(cluster_lists[r])
        cluster_coord = np.zeros((A.shape[0], A.shape[1], A.shape[2]))
        for s in range(len(cluster_indi)):
            cluster_coord[cluster_indi[s][0],cluster_indi[s][1],cluster_indi[s][2]] = 1
        cluster_nib = nib.Nifti1Image(cluster_coord, affine=np.eye(4))
        nib.save(cluster_nib, "DBSCAN-cluster_" + str(r) + ".nii.gz")

        filename_nii = "DBSCAN-cluster_" + str(r) + ".nii.gz"
            
        filename_stl = filename_nii[:-7] + '.stl'
        
        label = 1
            
        nii_2_mesh(filename_nii, filename_stl, label)
            
        shutil.move("DBSCAN-cluster_" + str(r) + ".nii.gz", 'Clusters_' + str(cluster_label))
        shutil.move("DBSCAN-cluster_" + str(r) + ".stl", 'Clusters_' + str(cluster_label))
        
    # Cluster files created and saved in 'Clusters_#' folder.


DBSCAN_Segmentation()
