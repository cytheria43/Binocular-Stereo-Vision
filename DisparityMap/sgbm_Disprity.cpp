#include <highgui.h>
#include <cv.h>
#include <cxcore.h>
#include <iostream>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include<math.h>
using namespace std;
using namespace cv;

const int imageWidth = 640;                             //摄像头的分辨率  
const int imageHeight = 480;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL,Obj_img;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat xyz;              //三维坐标

int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
StereoBM bm;
StereoSGBM sgbm;
Mat Camera_Left;
Mat distCoeff_Left  ;

Mat Camera_Right;
Mat distCoeff_Right;

Mat T;//T平移向量
Mat om;//om旋转向量
Mat R;//R 旋转矩阵

IplImage *L_cap;
IplImage *R_cap;

vector<KeyPoint> objectKeypoints, sceneKeypoints, keypoints1, keypoints2;
Mat objectDescriptors, sceneDescriptors, descriptros1, descriptros2;
vector<DMatch> obj_match, goodMatches;
vector<Point2f> scene_corners(4);
/****加载相机内外参数****/
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
	cout << Camera_Left << endl;
	cout << Camera_Right << endl;
	cout << distCoeff_Left << endl; 
	cout << distCoeff_Right << endl;

	cout << T << endl;
	cout << om << endl;
	cout << "摄像头参数加载成功！" << endl;
}
/******相机左右图像匹配******/
int sift_match(Mat img1,Mat img2)
{
	Mat image1 = img1; /*imread("temp1.jpg"); */ 

	Mat image2 = img2; /*imread("temp2.jpg");  */  //

	// 检测surf特征点

	//vector<KeyPoint> keypoints1, keypoints2;

	/*SurfFeatureDetector detector(400);*/
	SiftFeatureDetector detector(400);
	detector.detect(image1, keypoints1);

	detector.detect(image2, keypoints2);

	// 描述surf特征点

	//SurfDescriptorExtractor surfDesc;
	SiftDescriptorExtractor surfDesc;
	Mat descriptros1, descriptros2;

	surfDesc.compute(image1, keypoints1, descriptros1);

	surfDesc.compute(image2, keypoints2, descriptros2);

	// 计算匹配点数

	BruteForceMatcher<L2<float>>matcher;

	vector<DMatch> matches;

	matcher.match(descriptros1, descriptros2, matches);

	//std::nth_element(matches.begin(), matches.begin() + 24, matches.end());

	//matches.erase(matches.begin() + 25, matches.end());

	// 画出匹配图

	Mat imageMatches;

	drawMatches(image1, keypoints1, image2, keypoints2, matches,

		imageMatches);

	namedWindow("match");

	imshow("match", imageMatches);


	double max_dist = 0;

	double min_dist = 100;

	for (int i = 0; i<matches.size(); i++)

	{

		double dist = matches[i].distance;

		if (dist < min_dist) min_dist = dist;

		if (dist > max_dist) max_dist = dist;

	}

	cout << "最大距离：" << max_dist << endl;

	cout << "最小距离：" << min_dist << endl;



	//筛选出较好的匹配点  

	//vector<DMatch> goodMatches;

	for (int i = 0; i<matches.size(); i++)

	{

		if (matches[i].distance < 0.5 * max_dist)

		{

			goodMatches.push_back(matches[i]);

		}

	}

	cout << "goodMatch个数：" << goodMatches.size() << endl;



	//画出匹配结果  

	Mat img_matches;

	//红色连接的是匹配的特征点对，绿色是未匹配的特征点  

	drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, img_matches,

		Scalar::all(-1)/*CV_RGB(255,0,0)*/, CV_RGB(0, 255, 0), Mat(), 2);



	imshow("MatchSIFT", img_matches);

	return 1;

}

/******景深可分别采用SGBM和BM*****/
void stereo_match(int, void*)
{
	bm.state->roi1 = validROIL;
	bm.state->roi2 = validROIR;
	bm.state->preFilterCap = 31;
	bm.state->SADWindowSize = 15;
	bm.state->minDisparity = 0;
	bm.state->numberOfDisparities = 32;
	bm.state->textureThreshold = 10;
	bm.state->uniquenessRatio = 10;
	bm.state->speckleWindowSize = 100;
	bm.state->speckleRange = 32;
	bm.state->disp12MaxDiff = 1;

	Mat disp, disp8;

	//Mat img1border, img2border;
	//if (numDisparities != bm.state->numberOfDisparities)
	//	numDisparities = bm.state->numberOfDisparities;
	//copyMakeBorder(rectifyImageL, img1border, 0, 0, bm.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	//copyMakeBorder(rectifyImageR, img2border, 0, 0, bm.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);

	/*bm(rectifyImageL, rectifyImageR, disp);*/
	bm(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图

	

	/*disp = displf.colRange(bm.state->numberOfDisparities, rectifyImageL.cols);*/
	disp.convertTo(disp8, CV_8U, 255 / ((32 * 16 + 16)*16.));//计算出的视差是CV_16S格式
	reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;
	imshow("bm-disparity", disp8);
}
void stereo_match_sgbm(int, void*){
	int SADWindowSize = 13;
	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

	sgbm.fullDP = 1;

	int cn = 1;
	int numberOfDisparities = 256;
	sgbm.P1 = 8 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = 100;
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 1;

	Mat disp,n_disp, disp8;
	sgbm(rectifyImageL, rectifyImageR, n_disp);
	//归一化
	normalize(n_disp, disp, 0, 255, CV_MINMAX, CV_8U);

	disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
	reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;
	imshow("sgbm-disparity", disp8);

}
/*****物体匹配 定位位置******/
void sift_obj_match(Mat obj_img, Mat scene_img)
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
		if (obj_match[i].distance < 3 * min_dist)
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

	namedWindow("obj_scene_match", 0);

	imshow("obj_scene_match", obj_scene_Matches);
	//---------------------定位物体位置--------------------------
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < obj_good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(objectKeypoints[obj_good_matches[i].queryIdx].pt);
		scene.push_back(sceneKeypoints[obj_good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(img_object.cols, 0);
	obj_corners[2] = cvPoint(img_object.cols, img_object.rows);
	obj_corners[3] = cvPoint(0, img_object.rows);
	/*std::vector<Point2f> scene_corners(4);*/

	perspectiveTransform(obj_corners, scene_corners, H);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

	//-- Show detected matches
	namedWindow("Good Matches & Object detection", 0);
	imshow("Good Matches & Object detection", img_matches);

	//return 1;

}

int main()
{
	//---------------------------------------------------
	//---------------------------------------------------
	//CvCapture* capture2 = cvCreateCameraCapture(1);
	//CvCapture* capture1 = cvCreateCameraCapture(0);

	//cvNamedWindow("left", 1);
	//cvNamedWindow("right", 1);
	//IplImage* frame1;
	//IplImage* frame2;
	//while (1)
	//{

	//	frame2 = cvQueryFrame(capture2);
	//	if (!frame2) break;
	//	cvShowImage("right", frame2);

	//	frame1 = cvQueryFrame(capture1);
	//	if (!frame1) break;
	//	cvShowImage("left", frame1);


	//	char c = cvWaitKey(33);
	//	/*if (c == 33)match(frame2,frame1);*/
	//	if (c == 32){
	//		L_cap = frame1;
	//		R_cap = frame2;
	//		break;
	//	}
	//	if (c == 27){ break; }

	//}
	//-------------------------------------------------

	//加载数据
	Load_data();


	//立体校正
	Rodrigues(om, R); //Rodrigues变换
	
	stereoRectify(Camera_Left, distCoeff_Left, Camera_Right, distCoeff_Right, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
		-1, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(Camera_Left, distCoeff_Left, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(Camera_Right, distCoeff_Right, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

	cout << Rl << endl;
	cout << Rr << endl;
	cout << Pl << endl;
	cout << Pr << endl;
	cout << Q << endl;

	/*cout << mapLx << endl;
	cout << mapLy << endl;

	cout << mapRx << endl;
	cout << mapRy << endl;*/
	Obj_img = imread("cup1.jpg", CV_LOAD_IMAGE_COLOR);//输入样例图片

	Mat Scene_img = imread("right03.jpg", CV_LOAD_IMAGE_COLOR);  /****从相机截图，这里为了调试直接输入图片*****/

	rgbImageL = /*Mat(L_cap);*/imread("left01.jpg", CV_LOAD_IMAGE_COLOR);
	cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
	rgbImageR = /*Mat(R_cap);*/ imread("right01.jpg", CV_LOAD_IMAGE_COLOR);
	cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
	imshow("ImageL Before Rectify", grayImageL);
	imshow("ImageR Before Rectify", grayImageR);

	//经过remap之后，左右相机的图像已经共面并且行对准了

	remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR,0);
	remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR,0);

	//把校正结果显示出来
	Mat rgbRectifyImageL, rgbRectifyImageR;
	cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //伪彩色图
	cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);

	//单独显示
	//rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
	//rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
	imshow("ImageL After Rectify", rgbRectifyImageL);
	imshow("ImageR After Rectify", rgbRectifyImageR);

	//显示在同一张图上
	Mat canvas;
	double sf;
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC3);   //注意通道

	//左图像画到画布上
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
	resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
	//rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
	cout << "Painted ImageL" << endl;

	//右图像画到画布上
	canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
	resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	//rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
	cout << "Painted ImageR" << endl;

	//画上对应的线条
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
	imshow("rectified", canvas);


	sift_obj_match(Obj_img, rgbRectifyImageL);

	sift_match(rgbRectifyImageL,rgbRectifyImageR);
	//match
	/*计算景深*/
	stereo_match_sgbm(0, 0);
	/*find_bm();*/
	Point p;
	/*for (int i = 0; i < 100; i += 20){
		p.x = i; p.y = i;
		cout << "in world coordinate: " << xyz.at<Vec3f>(p) << endl;
	}*/
	//p.x = scene_corners[0].x+scene_corners[2].x;

	//p.y = scene_corners[0].y + scene_corners[2].y;
	//cout << "z=：" << xyz.at<Vec3f>(p) << endl;
	for (int i = 0; i < goodMatches.size(); i++)
	{
		p.x = keypoints1[goodMatches[i].queryIdx].pt.x;
		p.y = keypoints2[goodMatches[i].queryIdx].pt.y;
		cout << "第" << i << "个点的坐标为：" << xyz.at<Vec3f>(p) << endl;

	}
	waitKey(0);
	return 0;
}

