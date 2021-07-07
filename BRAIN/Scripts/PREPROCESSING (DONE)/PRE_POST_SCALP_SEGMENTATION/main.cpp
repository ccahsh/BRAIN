#include <stdio.h>
#include <iostream>
#include <fstream>
#include "dbscan.h"

#define MINIMUM_POINTS 40     // minimum number of cluster
#define EPSILON (8)  // distance for clustering, metre^2

void readBenchmarkData(vector<Point>& points)
{
    // load point cloud
    FILE *stream;
    stream = fopen ("valid_coord_DBSCANprep.dat","ra");

    unsigned int minpts, num_points, cluster, i = 0;
    double epsilon;
    fscanf(stream, "%u\n", &num_points);

    Point *p = (Point *)calloc(num_points, sizeof(Point));

    while (i < num_points)
    {
          fscanf(stream, "%f,%f,%f,%d\n", &(p[i].x), &(p[i].y), &(p[i].z), &cluster);
          p[i].clusterID = UNCLASSIFIED;
          points.push_back(p[i]);
          ++i;
    }

    free(p);
    fclose(stream);
}

void exportResults(vector<Point>& points, int num_points)
{
    int i = 0;
    
    ofstream myfile ("DBSCAN_raw.csv");
    
    if (myfile.is_open())
    {
      for(int i = 0; i < num_points; i++){
          myfile << points[i].x << " " ;
          myfile << points[i].y << " " ;
          myfile << points[i].z << " " ;
          myfile << points[i].clusterID << " " ;
      }
      myfile.close();
    }
    
}

int main()
{    
    vector<Point> points;

    // read point data
    readBenchmarkData(points);

    // constructor
    DBSCAN ds(MINIMUM_POINTS, EPSILON, points);

    // main loop
    ds.run();

    // result of DBSCAN algorithm    
    exportResults(ds.m_points, ds.getTotalPointSize());

    return 0;
}
