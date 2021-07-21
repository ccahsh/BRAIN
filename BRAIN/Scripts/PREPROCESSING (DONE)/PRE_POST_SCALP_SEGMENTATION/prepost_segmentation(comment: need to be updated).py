
from __future__ import division
import os
import re
import vtk
import sys
import csv
import math
import shutil
import random
import pickle
from random import random
import numpy as np
import pandas as pd
from sympy import *
import nibabel as nib
from pathlib import Path
from numpy import linalg
from sklearn import metrics
import plotly.express as px
import matplotlib.pyplot as plt
from collections import namedtuple
from nipype.interfaces import fsl
import plotly.graph_objects as go
from sklearn.cluster import KMeans
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


#Ellipsoid tool

class EllipsoidTool:

    def __init__(self):
        
        pass
    
    def getMinVolEllipse(self, P=None, tolerance=0.01):

        (N, d) = np.shape(P)
        d = float(d)

        Q = np.vstack([np.copy(P.T), np.ones(N)])
        QT = Q.T
        
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        center = np.dot(P.T, u)

        A = linalg.inv(
                       np.dot(P.T, np.dot(np.diag(u), P)) -
                       np.array([[a * b for b in center] for a in center])
                       ) / d

        U, s, rotation = linalg.svd(A)
        radii = 1.0/np.sqrt(s)
        
        return (center, radii, rotation)

#Convexhull 2D tool

Point = namedtuple('Point', 'x y')

class ConvexHull(object):
    _points = []
    _hull_points = []

    def __init__(self):
        pass

    def add(self, point):
        self._points.append(point)

    def _get_orientation(self, origin, p1, p2):
        difference = (
            ((p2.x - origin.x) * (p1.y - origin.y))
            - ((p1.x - origin.x) * (p2.y - origin.y))
        )

        return difference

    def compute_hull(self):
    
        points = self._points

        start = points[0]
        min_x = start.x
        for p in points[1:]:
            if p.x < min_x:
                min_x = p.x
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start:

            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1

            for p2 in points:
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction > 0:
                        far_point = p2

            self._hull_points.append(far_point)
            point = far_point

    def get_hull_points(self):
        if self._points and not self._hull_points:
            self.compute_hull()

        return self._hull_points

    def display(self):
        x = [p.x for p in self._points]
        y = [p.y for p in self._points]
        plt.plot(x, y, marker='D', linestyle='None')
        hx = [p.x for p in self._hull_points]
        hy = [p.y for p in self._hull_points]
        plt.plot(hx, hy)

        plt.title('Convex Hull')
        plt.show()
        
#plane equation for three given points

def equation_plane(p1, p2, p3):
    
    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]
    x2 = p2[0]
    y2 = p2[1]
    z2 = p2[2]
    x3 = p3[0]
    y3 = p3[1]
    z3 = p3[2]

    global a,b,c,d

    a = ((y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1))/((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    b = ((x3 - x1) * (z2 - z1) - (x2 - x1) * (z3 - z1))/((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    c = 1
    d = - a * x1 - b * y1 - c * z1


    
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
                if abs(AA[x][y][z]-BB[x][y][z] > MIN_thres) and BB[x][y][z] == 0:
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
        if len(cluster_indi) >= 8000:
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


    
    # Task 3 : K-Means Segmentation
    
    filename = input("\nType in the path to the NIFTI file you would like to apply kmeans to. Include nii.gz format for the NIFTI file. For example, if the file is in 'a' folder, type './a/file_name'.\n\n")
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
        if not os.path.exists('kmeans_'+str(c+1)):
            os.mkdir('kmeans_'+str(c+1))
        shutil.move("kmeans_" + str(c+1) + ".nii.gz",'kmeans_'+str(c+1))

    print("\nK-means segmentation completed.\n")

    
    
    # Task 4 : MNI Polygonalization
    
    
    scalp_input = input("Type in the path to scalp file you desire to polygonalize to MNI space from K-means cluster folder. Please include nii.gz format. If the file is in 'a' directory, type './a/file_name'.\n")
    MNI_input = input("Type in the reference MNI scalp file. Please include nii.gz format.\n")

    #loading MNI SCALP template

    m = nib.load(str(MNI_input))
    M = np.array(m.dataobj)

    #finding ellipsoid center for MNI SCALP template (symbol: O)

    P = []

    for x in range(M.shape[0]):
        for y in range(M.shape[1]):
            for z in range(M.shape[2]):
                if M[x][y][z] != 0:
                    P += [[x,y,z]]

    P = np.array(P)
    P_y = [v[1] for v in P]

    #selecting 25000 points due to limits in RAM.
    randomlist = random.sample(range(0, len(P)), 25000)

    Q = []

    for i in range(len(randomlist)):
        Q += [P[randomlist[i]]]

    Q = np.array(Q)

    ET = EllipsoidTool()

    (center, radii, rotation) = ET.getMinVolEllipse(Q, .01)

    #MNI centerpoint only has x and z coordinates as ellipsoid coordinate to keep respect of y-coordinate symmetry
    O = [center[0], np.average(P_y), center[2]]

    #loading kmeans
    a = nib.load(str(scalp_input))
    A = np.array(a.dataobj)

    #saving kmeans coordinates
    K_coord = []

    for x in range(A.shape[0]):
      for y in range(A.shape[1]):
        for z in range(A.shape[2]):
          if A[x][y][z] != 0:
            K_coord += [[x,y,z]]

    #finding different centers of cluster
    C = []

    C_x = [v[0] for v in K_coord]
    C_y = [v[1] for v in K_coord]
    C_z = [v[2] for v in K_coord]

    xx = (max(C_x)+min(C_x))/2
    yy = (max(C_y)+min(C_y))/2
    zz = (max(C_z)+min(C_z))/2

    C_middle = [xx,yy,zz]
    C_median = [np.median(C_x),np.median(C_y),np.median(C_z)]
    C_centroid = [np.average(C_x),np.average(C_y),np.average(C_z)]

    #assigning center of cluster to one of three
    C_center = C_middle #subject to change

    #finding distance vector between O and C_center
    dis = np.array(C_center) - np.array(O)

    #finding equation for plane tangent to ray(O, C_center) at C_center
    x, y, z = symbols('x y z', real=True)
    z =  x*(-dis[0]/dis[2])+y*(-dis[1]/dis[2])+((C_center[0]*dis[0]+C_center[1]*dis[1]+C_center[2]*dis[2])/dis[2])

    #collecting x and y coordinates for intersection between ray (O, K_coord) and tangent plane
    Intsect = {}
    Intsect_all = {}

    for i in range(len(K_coord)):
      #xi = O[0] + t*DIS["ray{0}".format(i)][0] or O[0] + t*(np.array(K_coord[i]) - np.array(O))[0]
      #yi = O[1] + t*DIS["ray{0}".format(i)][1] or O[1] + t*(np.array(K_coord[i]) - np.array(O))[1]
      #zi = O[2] + t*DIS["ray{0}".format(i)][2] or O[2] + t*(np.array(K_coord[i]) - np.array(O))[2]
      t = Symbol('t', real=True)
      #eqn = Eq(zi, xi*(-dis[0]/dis[2])+yi*(-dis[1]/dis[2])+((C_center[0]*dis[0]+C_center[1]*dis[1]+C_center[2]*dis[2])/dis[2]))
      eqn = Eq((O[2] + t*(np.array(K_coord[i]) - np.array(O))[2]), (O[0] + t*(np.array(K_coord[i]) - np.array(O))[0])*(-dis[0]/dis[2])+(O[1] + t*(np.array(K_coord[i]) - np.array(O))[1])*(-dis[1]/dis[2])+((C_center[0]*dis[0]+C_center[1]*dis[1]+C_center[2]*dis[2])/dis[2]))
      t_desired = (solve(eqn))[0]
      #saving x and y coordinates
      Intsect["set{0}".format(i)] = O[0] + t_desired*(np.array(K_coord[i]) - np.array(O))[0], O[1] + t_desired*(np.array(K_coord[i]) - np.array(O))[1]
      Intsect_all["set{0}".format(i)] = O[0] + t_desired*(np.array(K_coord[i]) - np.array(O))[0], O[1] + t_desired*(np.array(K_coord[i]) - np.array(O))[1], O[2] + t_desired*(np.array(K_coord[i]) - np.array(O))[2]

    #convexhull 2D
    Intersection_list = list(Intsect.values())

    #Intersection_list = list(Intersection.values())
    ch = ConvexHull()

    #saving coordinates of convexhull vertices
    for i in range(len(Intersection_list)):
      ch.add(Point(Intersection_list[i][0], Intersection_list[i][1]))
    A = ch.get_hull_points()
    Intersection_Vertices = []
    for i in range(len(A)):
      Intersection_Vertices += [[A[i][0],A[i][1]]]
    print(Intersection_Vertices)

    #finding equations for lateral sides of projected 'cone' from O to convexhull vertices, with up/down determined
    equations_updown = []

    for i in range(len(Intersection_Vertices) - 1):
      #finding first coordinate by key-value positions
      position1 = (list(Intsect.values())).index((Intersection_Vertices[i][0], Intersection_Vertices[i][1]))
      coordinates1 = list((list(Intsect_all.values()))[position1])

      #finding neighbouring (ascending order) coordinate by key-value positions
      position2 = (list(Intsect.values())).index((Intersection_Vertices[i+1][0], Intersection_Vertices[i+1][1]))
      coordinates2 = list((list(Intsect_all.values()))[position2])

      #finding plane
      equation_plane(O, coordinates1, coordinates2)

      #comparing z values to determine inequality sign (what if LHS = RHS? hope that doesn't happen...)
      if ((-a)*C_center[0] + (-b)*C_center[1] - d) > C_center[2]:
        e = -1
      if ((-a)*C_center[0] + (-b)*C_center[1] - d) < C_center[2]:
        e = 1
      equations_updown += [[-a, -b, -d, e]]

    #finalizing MNI scalp voxels
    segment_voxels = P.copy()

    for j in range(len(P)):
      for k in range(len(equations_updown)):
        if (segment_voxels[j] == P[j]).all():
          if equations_updown[k][3] == -1:
            if not (P[j][2] < equations_updown[k][0]*P[j][0] + equations_updown[k][1]*P[j][1] + equations_updown[k][2]):
              segment_voxels[j] = [10000,10000,10000]
          else:
            if not (P[j][2] > equations_updown[k][0]*P[j][0] + equations_updown[k][1]*P[j][1] + equations_updown[k][2]):
              segment_voxels[j] = [10000,10000,10000]
        else:
          break

    final = np.zeros((M.shape[0], M.shape[1], M.shape[2]))

    for i in range(len(segment_voxels)):
      if (segment_voxels[i] != [10000,10000,10000]).all():
        final[int(segment_voxels[i][0])][int(segment_voxels[i][1])][int(segment_voxels[i][2])] = 1

    MNI_segmented = nib.Nifti1Image(final, affine=np.eye(4))
    nib.save(MNI_segmented, str(scalp_input)[:-7] + "_MNI_version.nii.gz")

    print('Polygonalization completed.')


