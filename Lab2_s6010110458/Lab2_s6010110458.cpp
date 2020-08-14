// Lab2_s6010110458.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;


int MAX_KERNEL_LENGTH = 15;
int KERNEL_SIZE = 13;
float alpha = 0.4;
RNG rng(12345);

int main(int argc, char** argv)
{
	
	Mat capt, acc, background, motion,frame;
	VideoCapture vid("..\\video.avi");
	if (!vid.isOpened()) {
		cout << "couldn't open video." << endl;
		return -1;
	}
	vid.read(capt);
	acc = Mat::zeros(capt.size(), CV_32FC3);
	for (;;) {
		vid >> frame;
		if (capt.empty()) {
			//vid.release();
			cout << "Finished!" << endl;
			break;
		}
		//Week 2: Image Reprocessing
		
		//cvtColor(capt, capt, COLOR_BGR2GRAY);
		//equalizeHist(capt, capt);

		//Week 3.1 : Motion Tracking
		GaussianBlur(frame, capt, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0);
		accumulateWeighted(capt, acc, alpha);
		convertScaleAbs(acc, background);
		subtract(capt, background, motion);
		threshold(motion, capt, 20, 255, THRESH_BINARY);
		cvtColor(capt, capt, COLOR_BGR2GRAY);
		threshold(capt, capt, 10, 255, THRESH_BINARY);
	
		//morphologyEx(capt,capt,MORPH_OPEN,Size)
		
		
		//Week 3.2 : Contour
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(capt, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		/// Approximate contours to polygons + get bounding rects and circles
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());
		vector<Point2f>center(contours.size());
		vector<float>radius(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			minEnclosingCircle(contours_poly[i], center[i], radius[i]);
		}


		/// Draw polygonal contour + bonding rects + circles
		Mat drawing = frame;
		for (int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
		}
		
		

		//acc = (1-alpha) * frame + alpha * acc; <--- accumulateWeighted()
		
		
		imshow("motion detected", motion);
		imshow("motion threshold", capt);
		imshow("Contours", drawing);
		if (waitKey(30) >= 0) break;
	}
	
	return 0;
}

