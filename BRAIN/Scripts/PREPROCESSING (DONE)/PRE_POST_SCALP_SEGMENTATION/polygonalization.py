from __future__ import division
from sympy import *
import numpy as np
import nibabel as nib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from numpy import linalg
from random import random
import random
import pickle
from collections import namedtuple

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

#input scalp files

scalp_input = input("Type in the input scalp file. Please include nii.gz format.\n")
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
