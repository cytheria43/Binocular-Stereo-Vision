#include <highgui.h>
#include <cv.h>
#include <cxcore.h>
#include <iostream>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include<math.h>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

const int imageWidth = 640;                             //摄像头的分辨率  
const int imageHeight = 480;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL, Obj_img;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;//图像校正,裁剪之后的区域  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //映射查找表  
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q

Mat Camera_Left;
Mat distCoeff_Left;

Mat Camera_Right;
Mat distCoeff_Right;

Mat T;//T平移向量
Mat om;//om旋转向量
Mat R;//R 旋转矩阵

IplImage *L_cap;
IplImage *R_cap;

vector<KeyPoint> objectKeypoints, sceneKeypoints, keypoints1, keypoints2;
Mat objectDescriptors, sceneDescriptors, descriptros1, descriptros2;
vector<DMatch> obj_match, all_matches, goodMatches;
std::vector<Point2f> scene_corners(4);
float all_x = 0, all_y = 0; float r = 40;

void Load_data()
{
	CvMat *Intrinsics_Camera_Left = (CvMat *)cvLoad("Intrinsics_Camera_Left.xml");
	CvMat *Intrinsics_Camera_Right = (CvMat *)cvLoad("Intrinsics_Camera_Right.xml");
	CvMat *Distortion_Camera_Left = (CvMat *)cvLoad("Distortion_Camera_Left.xml");
	CvMat *Distortion_Camera_Right = (CvMat *)cvLoad("Distortion_Camera_Right.xml");
	CvMat *Translation = (CvMat *)cvLoad("Translation.xml");
	CvMat *RotRodrigues = (CvMat *)cvLoad("om.xml");
	/*CvMat *R_opencv = cvCreateMat(3, 3, CV_64F);
	cvRodrigues2(RotRodrigues_matlab, R_opencv);*/

	Camera_Left = Intrinsics_Camera_Left;
	distCoeff_Left = Distortion_Camera_Left;

	Camera_Right = Intrinsics_Camera_Right;
	distCoeff_Right = Distortion_Camera_Right;

	T = Translation;
	om = RotRodrigues;
	cout << "左摄像头内参数矩阵：" << Camera_Left << endl;
	cout << "右摄像头内参数矩阵：" << Camera_Right << endl;
	cout << "左摄像头畸变系数：" << distCoeff_Left << endl;
	cout << "右摄像头畸变系数：" << distCoeff_Right << endl;

	cout << "平移矩阵：" << T << endl;
	cout << "旋转矩阵：" << om << endl;
	cout << "摄像头参数加载成功！" << endl;
}

Mat sift_obj_match(Mat obj_img, Mat scene_img)
{
	Mat img_object = obj_img; /*imread("temp1.jpg"); */ //从数据库（文件夹）读入待寻找物体图片

	Mat img_scene = scene_img; /*imread("temp2.jpg");  */  //

	// 检测surf特征点

	//vector<KeyPoint> keypoints1, keypoints2;

	/*SurfFeatureDetector detector(400);*/
	SiftFeatureDetector detector(400);
	detector.detect(img_object, objectKeypoints);

	detector.detect(img_scene, sceneKeypoints);

	// 描述surf特征点

	//SurfDescriptorExtractor surfDesc;
	SiftDescriptorExtractor surfDesc;
	/*Mat descriptros1, descriptros2;*/

	surfDesc.compute(img_object, objectKeypoints, objectDescriptors);

	surfDesc.compute(img_scene, sceneKeypoints, sceneDescriptors);

	// 计算匹配点数

	FlannBasedMatcher matcher;

	/*vector<DMatch> matches;*/

	matcher.match(objectDescriptors, sceneDescriptors, obj_match);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < objectDescriptors.rows; i++)
	{
		double dist = obj_match[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	vector< DMatch > obj_good_matches;

	for (int i = 0; i < objectDescriptors.rows; i++)
	{
		if (obj_match[i].distance < 0.31 * max_dist)
		{
			obj_good_matches.push_back(obj_match[i]);
		}
	}

	Mat img_matches;
	drawMatches(img_object, objectKeypoints, img_scene, sceneKeypoints,
		obj_good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


	// 画出匹配图

	Mat obj_scene_Matches;

	drawMatches(img_object, objectKeypoints, img_scene, sceneKeypoints, obj_match,

		obj_scene_Matches);

	//namedWindow("obj_scene_match", 0);
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < obj_good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(objectKeypoints[obj_good_matches[i].queryIdx].pt);
		scene.push_back(sceneKeypoints[obj_good_matches[i].trainIdx].pt);
	}
	/*float all_x = 0, all_y = 0; float r=10;*/
	for (int t = 0; t < scene.size(); t++)
	{

		all_x += scene[t].x;
		all_y += scene[t].y;
	}
	all_x /= scene.size();
	all_y /= scene.size();
	//imshow("obj_scene_match", obj_scene_Matches);
	//---------------------定位物体位置--------------------------
	//std::vector<Point2f> obj;
	//
	//Mat H = findHomography(obj, scene, CV_RANSAC);

	////-- Get the corners from the image_1 ( the object to be "detected" )
	//std::vector<Point2f> obj_corners(4);
	//obj_corners[0] = cvPoint(0, 0);
	//obj_corners[1] = cvPoint(img_object.cols, 0);
	//obj_corners[2] = cvPoint(img_object.cols, img_object.rows);
	//obj_corners[3] = cvPoint(0, img_object.rows);
	///*std::vector<Point2f> scene_corners(4);*/

	//perspectiveTransform(obj_corners, scene_corners, H);

	////-- Draw lines between the corners (the mapped object in the scene - image_2 )
	//line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

	////-- Show detected matches
	namedWindow("Good Matches & Object detection", 0);
	imshow("Good Matches & Object detection", img_matches);
	return img_matches;
	//return 1;

}
void sift_scene_match(Mat img1, Mat img2)
{
	Mat image1 = img1; /*imread("temp1.jpg"); */ 
	Mat image2 = img2; /*imread("temp2.jpg");  */  //

	// 检测surf特征点

	//vector<KeyPoint> keypoints1, keypoints2;

	/*SurfFeatureDetector detector(400);*/
	SurfFeatureDetector detector(400);
	detector.detect(image1, keypoints1);

	detector.detect(image2, keypoints2);

	// 描述surf特征点

	//SurfDescriptorExtractor surfDesc;
	SurfDescriptorExtractor surfDesc;
	/*Mat descriptros1, descriptros2;*/

	surfDesc.compute(image1, keypoints1, descriptros1);

	surfDesc.compute(image2, keypoints2, descriptros2);

	// 计算匹配点数

	FlannBasedMatcher matcher;

	/*vector<DMatch> matches;*/

	matcher.match(descriptros1, descriptros2, all_matches);

	//std::nth_element(matches.begin(), matches.begin() + 24, matches.end());

	//matches.erase(matches.begin() + 25, matches.end());

	// 画出匹配图

	Mat imageMatches;

	drawMatches(image1, keypoints1, image2, keypoints2, all_matches,

		imageMatches);

	//namedWindow("match");

	//imshow("match", imageMatches);


	double max_dist = 0;

	double min_dist = 100;

	for (int i = 0; i<all_matches.size(); i++)

	{

		double dist = all_matches[i].distance;

		if (dist < min_dist) min_dist = dist;

		if (dist > max_dist) max_dist = dist;

	}

	//cout << "最大距离：" << max_dist << endl;

	//cout << "最小距离：" << min_dist << endl;



	//筛选出较好的匹配点  

	//vector<DMatch> goodMatches;

	for (int i = 0; i<all_matches.size(); i++)

	{

		if (all_matches[i].distance < 0.5 * max_dist)

		{

			goodMatches.push_back(all_matches[i]);

		}

	}

	cout << "goodMatch个数：" << goodMatches.size() << endl;



	//画出匹配结果  

	Mat img_matches;

	//红色连接的是匹配的特征点对，绿色是未匹配的特征点  

	drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, img_matches,

		Scalar::all(-1)/*CV_RGB(255,0,0)*/, CV_RGB(0, 255, 0), Mat(), 2);



	/*imshow("MatchSIFT", img_matches);*/

	/*return 1;*/

}



double mSift(Mat img,Mat img_match)
{
	//-----------------可用作截图实时更新物品图像库----------------------------------
	//---------------------------------------------------
	CvCapture* capture2 = cvCreateCameraCapture(2);
	CvCapture* capture1 = cvCreateCameraCapture(1);

	cvNamedWindow("left", 1);
	cvNamedWindow("right", 1);
	IplImage* frame1;
	IplImage* frame2;
	while (1)
	{

		frame2 = cvQueryFrame(capture2);
		if (!frame2){return -2; }
		cvShowImage("right", frame2);

		frame1 = cvQueryFrame(capture1);
		if (!frame1){return -1;}
		cvShowImage("left", frame1);


		char c = cvWaitKey(33);
		/*if (c == 33)match(frame2,frame1);*/
		if (c == 32){
			L_cap = frame1;
			R_cap = frame2;
			cvSaveImage("E:/bx/left.jpg", frame1);
			break;
		}
		if (c == 27){ break; }

	}
	//-------------------------------------------------

	//加载数据
	Load_data();
	//imshow("img",img);

	//立体校正
	Rodrigues(om, R); //Rodrigues变换

	stereoRectify(Camera_Left, distCoeff_Left, Camera_Right, distCoeff_Right, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
		-1, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(Camera_Left, distCoeff_Left, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(Camera_Right, distCoeff_Right, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
	
	//读取图片
	//Mat Scene_img = imread("right03.jpg", CV_LOAD_IMAGE_COLOR);

	rgbImageL = Mat(L_cap);//imread("E:\\camera\\left03.jpg"/*, CV_LOAD_IMAGE_COLOR*/);
	//bool e = rgbImageL.empty();
	/*if (e)
		return -1;*/
	cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
//	imshow("rgbl", rgbImageL);
	rgbImageR = Mat(R_cap); //imread("E:\\camera\\right03.jpg", CV_LOAD_IMAGE_COLOR);
	//if (rgbImageR.empty())return -2;
	cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

	//经过remap之后，左右相机的图像已经共面并且行对准了

	remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
	Mat rgbRectifyImageL, rgbRectifyImageR;
	cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //伪彩色图
	cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);
	
	imshow("ImageL After Rectify", rgbRectifyImageL);
	imshow("ImageR After Rectify", rgbRectifyImageR);
	/*sift_match(rgbImageL, rgbImageR);*/

	 sift_obj_match(img, rgbRectifyImageL);

	sift_scene_match(rectifyImageL, rgbRectifyImageR);

	vector< DMatch > correct_pair;  //左右摄像头匹配点在目标物体框内的keypoint集合
	for (int i = 0; i < all_matches.size(); i++)
	{
		if (abs(keypoints1[all_matches[i].queryIdx].pt.x-all_x)<r&&abs(keypoints1[all_matches[i].queryIdx].pt.y-all_y)<r)
		{
			if ((keypoints1[all_matches[i].queryIdx].pt.y-keypoints2[all_matches[i].trainIdx].pt.y)<10 )
			{
				correct_pair.push_back(all_matches[i]);
			}
			/*
			correct_pair.push_back(all_matches[i]);*/
		}

	}


	/********相似三角形测距法 计算距离*******/
	double B = sqrt(T.at<double>(0, 0)*T.at<double>(0, 0) + T.at<double>(1, 0)*T.at<double>(1, 0) + T.at<double>(2, 0)*T.at<double>(2, 0));
	double f = Q.at<double>(2, 3); /*678.5*/
	double avr_dis = 0;
	double max_dis = 0, min_dis = 666666;
	for (int i = 0; i < correct_pair.size(); i++)
	{

		double x_L = keypoints1[correct_pair[i].queryIdx].pt.x;
		double x_R = keypoints1[correct_pair[i].trainIdx].pt.x;

		double distance = f*B / abs(x_L - x_R);
		avr_dis += distance;

		if (distance < min_dis)min_dis = distance;
		if (distance>max_dis)max_dis = distance;

	}
	double finaldis=(double)((avr_dis - max_dis - min_dis) / (correct_pair.size() - 2));
	finaldis /= 1000;
	return finaldis;
	/*return 1000;*/
}



