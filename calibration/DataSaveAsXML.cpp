//#include "stdafx.h"  
#include <cv.h>  
#include <highgui.h>  
double dataL[9] = { 2093.73044, 0, 551.08227,
0, 2109.56737, 706.67736,
0, 0, 1
};
double dataR[9] = { 2117.28784, 0, 641.21264,
0, 2128.74984, 666.84360,
0, 0, 1
};


int main()
{
	double *data;
	data = dataL;
	CvMat intrinsic_matrix;

	cvInitMatHeader(&intrinsic_matrix, 3, 3, CV_64F, data);

	cvSave("intrinsics_LeftCamera.xml", &intrinsic_matrix);


	data = dataR;

	cvInitMatHeader(&intrinsic_matrix, 3, 3, CV_64F, data);

	cvSave("intrinsics_RightCamera.xml", &intrinsic_matrix);
	return 0;
}