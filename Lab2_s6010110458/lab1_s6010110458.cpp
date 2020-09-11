// Lab2_s6010110458.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "TrackingObjectSet.h"
#include "Tracker.h"
#include <opencv2/videoio.hpp>


using namespace cv;
using namespace std;


int MAX_KERNEL_LENGTH = 15;
int KERNEL_SIZE = 13;
float alpha = 0.4;
RNG rng(12345);
Mat morpho_kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
Mat morpho_kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
Tracker tracker = Tracker();

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
	int totalUp = 0;
	int totalDown = 0;
	for (;;) {
		vid >> frame;
		if (frame.empty()) {
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
		threshold(capt, capt, 5, 255, THRESH_BINARY);
		//ode(capt, capt, morpho_kernel1, Point(4, 4), 1);
		morphologyEx(capt, capt, MORPH_OPEN, morpho_kernel);
		dilate(capt, capt, morpho_kernel, Point(3,3), 2);
		
		
		//Week 3.2 : Contour
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(capt, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		/// Approximate contours to polygons + get bounding rects and circles
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());
		vector<Point2f>center(contours.size());
		vector<float>radius(contours.size());
		vector<Rect> classifiedRect;

		for (int i = 0; i < contours.size(); i++)
		{
			
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			if ((boundRect[i].height > 60 && boundRect[i].width > 80) && (boundRect[i].height < 200 && boundRect[i].width < 200))
				classifiedRect.push_back(boundRect[i]);
			minEnclosingCircle(contours_poly[i], center[i], radius[i]);
		}

		Mat drawing = frame;
	
		line(drawing, Point(0, drawing.rows / 2), Point(frame.cols, frame.rows / 2), Scalar(0, 0, 0), 2);
		// Draw polygonal contour + bonding rects + circles
		
		for (int i = 0; i < classifiedRect.size(); i++)
		{
			Scalar color = tracker.Track(classifiedRect[i],drawing.rows);
	
			//drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			rectangle(drawing, classifiedRect[i].tl(), classifiedRect[i].br(),color , 2, 8, 0);
			//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
			int begin = classifiedRect[i].y;
			int end = begin + classifiedRect[i].height;
			if ((begin < (drawing.rows / 2)) && (end > (drawing.rows / 2))) {
				int up_or_down = tracker.rectCounter(classifiedRect[i]);
				if (up_or_down == 1) {
					totalUp++;
				}
				else if (up_or_down == -1) {
					totalDown++;
				}
			}
		}
		cout << tracker.objsSet->objs.size() << endl;
		putText(drawing, "Total up: ", Point(20, 425), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(10, 255, 255), 2, 2);
		putText(drawing, to_string(totalUp), Point(150, 425), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(10, 255, 255), 2, 2);
		putText(drawing, "Total down: ", Point(20, 455), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(10, 255, 255), 2, 2);
		putText(drawing, to_string(totalDown), Point(180, 455), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(10, 255, 255), 2, 2);

		//acc = (1-alpha) * frame + alpha * acc; <--- accumulateWeighted()
		
		
		imshow("motion detected", motion);
		imshow("motion threshold", capt);
		imshow("Output", drawing);
		if (waitKey(10) >= 0) break;
	}
	
	return 0;
}


