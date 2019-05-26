#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
using namespace cv;
using namespace std;
IplImage* c1;
IplImage* c2;
const char* path;
const char* path1;

int main(int argc, char** argv)
{
	CvCapture* capture2 = cvCreateCameraCapture(1);
	CvCapture* capture1 = cvCreateCameraCapture(0);

	cvNamedWindow("lf", 1);
	cvNamedWindow("ri", 1);
	IplImage* frame1;
	IplImage* frame2;

	/*path = "D:\\data\\%d.jpg";
	path1 = "D:\\data\\right1.jpg";*/
	int idx = 0;
	int idx1 = 0;
	char adr[128] = { 0 };
	char adr1[128] = { 0 };
	while (1)
	{

		frame2 = cvQueryFrame(capture2);
		if (!frame2) break;
		cvShowImage("ri", frame2);

		frame1 = cvQueryFrame(capture1);
		if (!frame1) break;
		cvShowImage("lf", frame1);


		char c = cvWaitKey(33);
		/*if (c == 33)match(frame2,frame1);*/
		if (c == 32){ 
			c1 = frame1;
			c2 = frame2;
			sprintf(adr, "D:/data/0309/left%d.jpg", ++idx);
			sprintf(adr1, "D:/data/0309/right%d.jpg", ++idx1);
			cvSaveImage(adr, frame1);
			cvSaveImage(adr1, frame2);
			
		}
		if (c == 27){ break; }
		
	}
	/*match(c1, c2);*/
	cvReleaseCapture(&capture1); cvReleaseCapture(&capture2);
	cvDestroyWindow("lf"); cvDestroyWindow("ri");
}

