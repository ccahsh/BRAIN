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
from sklearn.cluster import KMeans
from nipype.testing import example_data
from mpl_toolkits.mplot3d import Axes3D


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
    
    filename = input("\nType in the name of NIFTI file you would like to apply kmeans to. Include nii.gz format.\n\n")
    clusters = input("\nType in the number of clusters you would like to segment in the file. This depends on the file, but usually it is 2 or 3, given that DBSCAN analysis has been done.\n\n")
    print("\nWorking...\n")
    
    a = nib.load(str(filename))
    A = np.array(a.dataobj)
    
    points = []
    
for x in range(A.shape[0]):
    for y in range(A.shape[1]):
        for z in range(A.shape[2]):
            if A[x][y][z] == 1:
                points += [[x,y,z]]
    
points = np.array(points)

kmeans = KMeans(n_clusters=int(clusters), random_state=1).fit(points)

compiled = [[] for _ in range(int(clusters))]

for l in range(len(kmeans.labels_)):
    compiled[int(kmeans.labels_[l])] += [points[l]]

for c in range(int(clusters)):
    i = np.zeros((A.shape[0],A.shape[1],A.shape[2]))
    for p in range(len(compiled[c])):
        i[compiled[c][p][0],compiled[c][p][1],compiled[c][p][2]] = 1
    I = nib.Nifti1Image(i, affine=np.eye(4))
    nib.save(I, "kmeans_" + str(c+1) + ".nii.gz")
    filename_nii = "kmeans_" + str(c+1) + ".nii.gz"
    filename_stl = filename_nii[:-7] + ".stl"
    label = 1
    nii_2_mesh(filename_nii,filename_stl,label)
    if not os.path.exists('kmeans_'+str(c+1)):
        os.mkdir('kmeans_'+str(c+1))
    shutil.move("kmeans_" + str(c+1) + ".nii.gz",'kmeans_'+str(c+1))
    shutil.move("kmeans_" + str(c+1) + ".stl",'kmeans_'+str(c+1))
    
print("\nK-means segmentation completed.\n")

