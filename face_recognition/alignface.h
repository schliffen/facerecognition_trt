//
// Created by ali on 7.01.2021.
//
#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H


//#include <LinearAlgebra.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"



typedef float fp_t;
typedef float dt_p;

struct ref_landmark{
    fp_t org_landmark_1[5][2] = {
//            {54.796, 49.99 },
//            {60.771, 50.115},
//            {76.673, 69.007},
//            {55.388, 89.702},
//            {61.257, 89.05 }
            {30.2946, 51.6963},
            {65.5318, 51.5014},
            {48.0252, 71.7366},
            {33.5493, 92.3655},
            {62.7299, 92.2041}
    };



    fp_t ref_mean[2] = {48.02616, 71.90078};
    fp_t std = 13.637027193284474;

    fp_t standardized[5][2] = {
            {-1.7731562e+01, -2.0204479e+01},
            { 1.7505638e+01, -2.0399380e+01},
            {-9.6130371e-04, -1.6417694e-01},
            {-1.4476860e+01,  2.0464722e+01},
            { 1.4703739e+01,  2.0303322e+01}
    };

};


bool transformation_matrix( fp_t flandmarks[10],fp_t M[3][3]);
bool invert_indx( int indi, int indj, fp_t rindex[2], fp_t M[3][3]);
bool alignface( unsigned char * src, int srcWidth, int srcHeight, std::vector<uint8_t>& templ, int dstWidth, int dstHeight, fp_t facelandmarks[10]);




cv::Mat alignface( cv::Mat src, int srcWidth, int srcHeight, int dstWidth, int dstHeight, fp_t facelandmarks[10]);


#endif //FR_NIST_UBUNTU_TRANSFORMATION_H
