//
// Created by ali on 7.01.2021.
//

#include "alignface.h"


#define SMALL_NUM 1e-12

ref_landmark reference_landmark;








bool transformation_matrix( dt_p flandmarks[10],fp_t M[3][3]){

    // calculate mean and variance of 5 landmarks
    dt_p fl_mean[2]; dt_p fl_var;
    fl_mean[0] = (flandmarks[0] + flandmarks[2] + flandmarks[4] + flandmarks[6] + flandmarks[8])/5.;
    fl_mean[1] = (flandmarks[1] + flandmarks[3] + flandmarks[5] + flandmarks[7] + flandmarks[9])/5.;
    // normalization
    dt_p mmx=0, mmy=0;
    for (int i=0; i<5; i++){
        flandmarks[2*i] -= fl_mean[0];
        flandmarks[2*i+1] -= fl_mean[1];

        mmx += flandmarks[2*i]/5;
        mmy += flandmarks[2*i+1]/5;
    }
    fl_var = (pow(flandmarks[0] - mmx,2) + pow(flandmarks[1] - mmy,2) + pow(flandmarks[2] - mmx,2)
              + pow(flandmarks[3] - mmy,2) + pow(flandmarks[4] - mmx,2) + pow(flandmarks[5] - mmy,2) + pow(flandmarks[6] - mmx,2)
              + pow(flandmarks[7] - mmy,2) + pow(flandmarks[8] - mmx,2) + pow(flandmarks[9] - mmy,2)) /5 ;

    // matrix multiplication part (srd[2x5] x dst[5x2])
    dt_p matrixpxp [4] = {0,0,0,0};
    for (int i=0; i<5;i++) {
        matrixpxp[0] += (1 / 5.) * flandmarks[2*i] * reference_landmark.standardized[i][0];
        matrixpxp[1] += (1. / 5.) * flandmarks[2*i+1] * reference_landmark.standardized[i][0];
        matrixpxp[2] += (1. / 5.) * flandmarks[2*i] * reference_landmark.standardized[i][1];
        matrixpxp[3] += (1. / 5.) * flandmarks[2*i+1] * reference_landmark.standardized[i][1];
    }
    // compute determinant of A
    dt_p detA = matrixpxp[0] * matrixpxp[3] - matrixpxp[2] * matrixpxp[1];
    dt_p d[2] = {1,1};
    int rankA;
    if (detA<0)
        d[1] = -1;
    if (detA < SMALL_NUM || -detA > -1*SMALL_NUM)
        rankA = (matrixpxp[0] == 0 || matrixpxp[1]==0 || matrixpxp[2]==0 || matrixpxp[3]==0) ? 0:1;
    else
        rankA = 2;

    if (rankA==0)
        return false;

    // computing svd
    //a, b, c ,d = Mat[0], Mat[1], Mat[3], Mat[4]
    dt_p theta = .5 * std::atan2( 2*matrixpxp[0]*matrixpxp[2] + 2*matrixpxp[1]*matrixpxp[3], pow(matrixpxp[0],2) + pow(matrixpxp[1],2) - pow(matrixpxp[2],2) - pow(matrixpxp[3],2));
    dt_p psi = .5 * std::atan2(2*matrixpxp[0]*matrixpxp[1] + 2*matrixpxp[2]*matrixpxp[3], pow(matrixpxp[0],2) - pow(matrixpxp[1],2) + pow(matrixpxp[2],2) - pow(matrixpxp[3],2));

    dt_p U[4];
    U[0] = cos(theta); U[1] = std::sin(theta), U[2] = std::sin(theta), U[3] = -std::cos(theta);

    dt_p S1 = pow(matrixpxp[0],2) + pow(matrixpxp[1],2) + pow(matrixpxp[2],2) + pow(matrixpxp[3],2);

    dt_p S2 = std::sqrt(pow(pow(matrixpxp[0],2) + pow(matrixpxp[1],2) - pow(matrixpxp[2],2) - pow(matrixpxp[3],2), 2) + 4*pow(matrixpxp[0]*matrixpxp[2] + matrixpxp[1]*matrixpxp[3], 2));
    //
    dt_p S[2];
    S[0] = std::sqrt((S1 + S2)/2);
    S[1] = std::sqrt((S1 - S2)/2);
//    S[0] = {sig1, sig2}

    dt_p s11 = (matrixpxp[0]*std::cos(theta) + matrixpxp[2]*std::sin(theta))*std::cos(psi) + (matrixpxp[1]*std::cos(theta) + matrixpxp[3]*std::sin(theta))*std::sin(psi);
    dt_p s22 = (matrixpxp[0]*std::sin(theta) + matrixpxp[2]*std::cos(theta))*std::sin(psi) + (-matrixpxp[1]*std::sin(theta) + matrixpxp[3]*std::cos(theta))*std::cos(psi);

    float Vt [4];
    Vt[0] = s11/std::abs(s11) * std::cos(psi); Vt[1] = s11/std::abs(s11) * std::sin(psi);
    Vt[2] = s22/std::abs(s22) * std::sin(psi); Vt[3] = -s22/std::abs(s22) * std::cos(psi);

    // conreolling the rank of the matrix
    // in case of low rank matrices
    if (rankA==1){
        dt_p detU = U[0]*U[3] - U[1]*U[2];
        dt_p detV = Vt[0]*U[3] - U[1]*U[2];
        if (detU * detV == 0){
            M[0][0] = U[0]*Vt[0] + U[1]*Vt[1];
            M[0][1] = U[0]*Vt[2] + U[1]*Vt[3];
            M[1][0] = U[2]*Vt[0] + U[3]*Vt[1];
            M[1][1] = U[2]*Vt[2] + U[3]*Vt[3];
            M[0][2] = 0;
            M[1][2] = 0;
        } else{
            uint8_t tmps = d[1];
            d[1] = -1;
            M[0][0] = d[0]*U[0]*Vt[0] + d[1]*U[1]*Vt[1];
            M[0][1] = d[0]*U[0]*Vt[2] + d[1]*U[1]*Vt[3];
            M[1][0] = d[0]*U[2]*Vt[0] + d[1]*U[3]*Vt[1];
            M[1][1] = d[0]*U[2]*Vt[2] + d[1]*U[3]*Vt[3];
            M[0][2] = 0;
            M[1][2] = 0;
            d[1] = tmps;
        }
    }else{
        M[0][0] = d[0]*U[0]*Vt[0] + d[1]*U[1]*Vt[1];
        M[0][1] = d[0]*U[0]*Vt[2] + d[1]*U[1]*Vt[3];
        M[1][0] = d[0]*U[2]*Vt[0] + d[1]*U[3]*Vt[1];
        M[1][1] = d[0]*U[2]*Vt[2] + d[1]*U[3]*Vt[3];
        M[0][2] = 0;
        M[1][2] = 0;
    }

    // scale is true
    //if (estimate_scale):
    // Eq. (41) and (42).
    // compute S x d
    dt_p scale = 1.0 / fl_var * (S[0]* d[0] + S[1]* d[1]);
    //
    M[0][2] = reference_landmark.ref_mean[0] - scale * (M[0][0] * fl_mean[0] + M[0][1] * fl_mean[1]);
    M[1][2] = reference_landmark.ref_mean[1] - scale * (M[1][0] * fl_mean[0] + M[1][1] * fl_mean[1]) ;
    //
    M[0][0] *= scale;
    M[0][1] *= scale;
    M[1][0] *= scale;
    M[1][1] *= scale;

    M[2][0] = 0; M[2][1] = 0; M[2][2] = 1.;
}

bool invert_indx( int indi, int indj, dt_p rindex[2], dt_p M[3][3]){
    //
    // get inverse of 2d matrix
    dt_p iindex[2];
    iindex[0] = indi - M[0][2]; // translation
    iindex[1] = indj - M[1][2]; //
    dt_p det = 1/ (M[0][0]*M[1][1] - M[0][1]*M[1][0]);
    if (det < 1e-9){
        std::cout<< " determinant is close to zero \n";
        return false;}
    // inversion
    rindex[1] = det * (iindex[0]*M[1][1] - iindex[1]*M[0][1]);
    rindex[0] = det * (-iindex[0]*M[1][0] + iindex[1]*M[0][0]);

    return true;
}



cv::Mat alignface( cv::Mat src, int srcWidth, int srcHeight, int dstWidth, int dstHeight, fp_t facelandmarks[10]){
    // the code shoul be modified
    // get transformation matrix
    cv::Mat warp_dst;
    
    // wrap alignment with opencv 
    cv::Point2f landmark_cv[5];
    landmark_cv[0] = cv::Point2f(30.2946, 51.6963);
    landmark_cv[1] = cv::Point2f(65.5318, 51.5014);
    landmark_cv[2] = cv::Point2f(48.0252, 71.7366);
    landmark_cv[3] = cv::Point2f(33.5493, 92.3655);
    landmark_cv[4] = cv::Point2f(62.7299, 92.2041);

    // adapting transformation matrix
    cv::Point2f lndmrks[5];
    lndmrks[0] = cv::Point2f(facelandmarks[0], facelandmarks[1]);
    lndmrks[1] = cv::Point2f(facelandmarks[2], facelandmarks[3]); 
    lndmrks[2] = cv::Point2f(facelandmarks[4], facelandmarks[5]); 
    lndmrks[3] = cv::Point2f(facelandmarks[5], facelandmarks[7]); 
    lndmrks[4] = cv::Point2f(facelandmarks[8], facelandmarks[9]); 
    
    // ger wrap matrix
    // Mat warp_dst = Mat::zeros( src.rows, src.cols, src.type() );
    cv::Mat warp_mat = getAffineTransform( lndmrks, landmark_cv );

    cv::warpAffine( src, warp_dst, warp_mat, warp_dst.size() ); 

    return warp_dst(cv::Range(0, 112), cv::Range(0, 112));
}

bool alignface( unsigned char * src, int srcWidth, int srcHeight, std::vector<uint8_t>& templ, int dstWidth, int dstHeight, fp_t facelandmarks[10]){
    // the code shoul be modified
    // get transformation matrix
    templ.clear();
    fp_t transformation_mat[3][3];
    transformation_matrix( facelandmarks, transformation_mat);
    // inverting index
    dt_p rindex[2];
    //
    int scale = 1;
    dt_p b1234[4] = {0,0,0,0};
    // put the starting point on Pmin
    //
    for (int i = 0; i < dstHeight; i++) {
        // go to the next line
        for (int j = 0; j < dstWidth; j++) {
            scale = 1;
            // aligment and allocation at the same time
            invert_indx( j ,i , rindex, transformation_mat);
            // get neigbours
            int32_t neighbour[4] = {0,0,0,0};
            if (std::floor(rindex[0])>=0 && std::ceil(rindex[0])< srcHeight && std::floor(rindex[1])>=0 && std::ceil(rindex[1])< srcWidth ){
                neighbour[0] = std::floor(rindex[0]);
                neighbour[1] = std::ceil(rindex[0]);
                neighbour[2] = std::floor(rindex[1]);
                neighbour[3] = std::ceil(rindex[1]);

            }else{

                if (rindex[0]<0){
                    neighbour[0] = 0;
                    neighbour[1] = 1;
                } else if (rindex[0]>srcHeight){
                    neighbour[0] = srcHeight-2;
                    neighbour[1] = srcHeight-1;
                }

                if (rindex[1]<0){
                    neighbour[2] = 0;
                    neighbour[3] = 1;
                } else if (rindex[1]>srcWidth){
                    neighbour[2] = srcWidth-2;
                    neighbour[3] = srcWidth-1;
                }

            }
            // to modify this
            for (int c = 0; c < 3; c++) // dst-- for GBR version
            {
                b1234[0] =  *(src + 3*(neighbour[0]* srcWidth +  neighbour[2]) + c ); // float(img[neighbour[0], neighbour[2]]);
                b1234[1] =  *( src + 3*(neighbour[1]*srcWidth + neighbour[2]) + c) - *( src + 3*(srcWidth*neighbour[0] + neighbour[2]) + c);
                b1234[2] =  *( src + 3*(neighbour[0]*srcWidth + neighbour[3]) + c) - *( src + 3*(srcWidth*neighbour[0] + neighbour[2]) + c);
                b1234[3] =  *( src + 3*(neighbour[0]*srcWidth + neighbour[2]) + c) - *( src + 3*(srcWidth*neighbour[1] + neighbour[2]) + c) -
                            *( src + 3*(neighbour[0]*srcWidth + neighbour[3]) + c) + *( src + 3*(srcWidth*neighbour[1] + neighbour[3]) + c);

                int32_t intensity =  b1234[0] + b1234[1]*(rindex[0]-neighbour[0])*(neighbour[3]-rindex[1]) + b1234[2]*(rindex[1]-neighbour[2])*(neighbour[1]-rindex[0]) +
                                     b1234[3]*(rindex[0]-neighbour[0])*(rindex[1] - neighbour[2]) ;
                intensity = intensity >= 0 ?  intensity : 0;
                intensity = intensity <= 255 ?  intensity : 255;
//                std::cout<< "intensity rescaled: " << (intensity/255 - .5)/.5 << std::endl;
                templ.push_back(intensity);
            }
        }
    }

//templ.data() = dst;

}


