// circleDetector.cpp
//
// Kevin Hughes 
//
// 2012
//
// This is an implementation of the circle detction RANSAC algorithm
// described in "An efficient randomized algorithm for detecting circles"
// by Chen, T.C. and Chung, K.L.
//
 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <vector>
#include <string>

#include <time.h>

using namespace cv;
using namespace std;

// circleRANSAC
//
// input:
//		image - either CV_8UC1 or CV_8UC3
//		circles - return vector of Vec3f (x,y,radius)
//		canny_threshold - higher canny threshold, lower is set to canny_threshold / 2
//		circle_threshold - value between 0 and 1 for the percentage of the circle that needs to vote for it to be accepted
//		numIterations - the number of RANSAC loops, the function will quit if there is no points left in the set
//
void circleRANSAC(Mat &image, vector<Vec3f> &circles, double canny_threshold, double circle_threshold, int numIterations);

int main(int argc, char *argv[])
{
	if(argc != 5)
	{
		cout << "Usage: " << argv[0] << "<image file> <canny threshold> <circle threshold> <iterations>" << endl;
		return -1;
	}
	
	// collect arguemtns
	string filename = argv[1];
	double canny_threshold = atof(argv[2]);
	double circle_threshold = atof(argv[3]);
	int iterations = atoi(argv[4]);
	
	Mat image = imread(filename,0);
	vector<Vec3f> circles;
	
	const clock_t start = clock();
	circleRANSAC(image, circles, canny_threshold, circle_threshold, iterations);
	clock_t end = clock();
	
	cout << "Found " << (int)circles.size() << " Circles." << endl;
	
	double time = ((double)(end - start)) / (double)CLOCKS_PER_SEC;
	std::cout << "RANSAC runtime: " << time << " seconds" << std::endl;
	
	// Draw Circles
	cvtColor(image,image,CV_GRAY2RGB);
	for(int i = 0; i < (int)circles.size(); i++)
	{
		int x = circles[i][0];
		int y = circles[i][1];
		float rad = circles[i][2];
		
		circle(image, Point(x,y), rad, Scalar(0,255,0));
	}
	
	imshow("circles", image);
	waitKey();
	
	return 0;
}

void circleRANSAC(Mat &image, std::vector<Vec3f> &circles, double canny_threshold, double circle_threshold, int numIterations)
{
	CV_Assert(image.type() == CV_8UC1 || image.type() == CV_8UC3);
	circles.clear();
	
	// Edge Detection
	Mat edges;
	Canny(image, edges, MAX(canny_threshold/2,1), canny_threshold, 3);
	
	// Create point set from Canny Output
	std::vector<Point2d> points;
	for(int r = 0; r < edges.rows; r++)
	{
		for(int c = 0; c < edges.cols; c++)
		{
			if(edges.at<unsigned char>(r,c) == 255)
			{
				points.push_back(cv::Point2d(c,r));
			}
		}	
	}
	
	// 4 point objects to hold the random samples
	Point2d pointA;
	Point2d pointB;
	Point2d pointC;
	Point2d pointD;
	
	// distances between points
	double AB;
	double BC;
	double CA;
	double DC;

	// varibales for line equations y = mx + b
	double m_AB;
	double b_AB;
	double m_BC;
	double b_BC;

	// varibles for line midpoints
	double XmidPoint_AB;
	double YmidPoint_AB;
	double XmidPoint_BC;
	double YmidPoint_BC;

	// variables for perpendicular bisectors
	double m2_AB;
	double m2_BC;
	double b2_AB;
	double b2_BC;

	// RANSAC
	cv::RNG rng; 
	int min_point_separation = 10; // change to be relative to image size?
	int colinear_tolerance = 1; // make sure points are not on a line
	int radius_tolerance = 3; // change to be relative to image size?
	int points_threshold = 10; //should always be greater than 4
	//double min_circle_separation = 10; //reject a circle if it is too close to a previously found circle
	//double min_radius = 10.0; //minimum radius for a circle to not be rejected
	
	int x,y;
	Point2d center;
	double radius;
	
	// Iterate
	for(int iteration = 0; iteration < numIterations; iteration++) 
	{
		//std::cout << "RANSAC iteration: " << iteration << std::endl;
		
		// get 4 random points
		pointA = points[rng.uniform((int)0, (int)points.size())];
		pointB = points[rng.uniform((int)0, (int)points.size())];
		pointC = points[rng.uniform((int)0, (int)points.size())];
		pointD = points[rng.uniform((int)0, (int)points.size())];
		
		// calc lines
		AB = norm(pointA - pointB);
		BC = norm(pointB - pointC);
		CA = norm(pointC - pointA);
		DC = norm(pointD - pointC);
		
		// one or more random points are too close together
		if(AB < min_point_separation || BC < min_point_separation || CA < min_point_separation || DC < min_point_separation) continue;
		
		//find line equations for AB and BC
		//AB
		m_AB = (pointB.y - pointA.y) / (pointB.x - pointA.x + 0.000000001); //avoid divide by 0
		b_AB = pointB.y - m_AB*pointB.x;

		//BC
		m_BC = (pointC.y - pointB.y) / (pointC.x - pointB.x + 0.000000001); //avoid divide by 0
		b_BC = pointC.y - m_BC*pointC.x;
		
		
		//test colinearity (ie the points are not all on the same line)
		if(abs(pointC.y - (m_AB*pointC.x + b_AB + colinear_tolerance)) < colinear_tolerance) continue;
		
		//find perpendicular bisector
		//AB
		//midpoint
		XmidPoint_AB = (pointB.x + pointA.x) / 2.0;
		YmidPoint_AB = m_AB * XmidPoint_AB + b_AB;
		//perpendicular slope
		m2_AB = -1.0 / m_AB;
		//find b2
		b2_AB = YmidPoint_AB - m2_AB*XmidPoint_AB;

		//BC
		//midpoint
		XmidPoint_BC = (pointC.x + pointB.x) / 2.0;
		YmidPoint_BC = m_BC * XmidPoint_BC + b_BC;
		//perpendicular slope
		m2_BC = -1.0 / m_BC;
		//find b2
		b2_BC = YmidPoint_BC - m2_BC*XmidPoint_BC;
		
		//find intersection = circle center
		x = (b2_AB - b2_BC) / (m2_BC - m2_AB);
		y = m2_AB * x + b2_AB;	
		center = Point2d(x,y);
		radius = cv::norm(center - pointB);
		
		/// geometry debug image
		if(false)
		{
			Mat debug_image = edges.clone();
			cvtColor(debug_image, debug_image, CV_GRAY2RGB);
		
			Scalar pink(255,0,255);
			Scalar blue(255,0,0);
			Scalar green(0,255,0);
			Scalar yellow(0,255,255);
			Scalar red(0,0,255);
		
			// the 3 points from which the circle is calculated in pink
			circle(debug_image, pointA, 3, pink);
			circle(debug_image, pointB, 3, pink);
			circle(debug_image, pointC, 3, pink);
		
			// the 2 lines (blue) and the perpendicular bisectors (green)
			line(debug_image,pointA,pointB,blue);
			line(debug_image,pointB,pointC,blue);
			line(debug_image,Point(XmidPoint_AB,YmidPoint_AB),center,green);
			line(debug_image,Point(XmidPoint_BC,YmidPoint_BC),center,green);
		
			circle(debug_image, center, 3, yellow); // center
			circle(debug_image, center, radius, yellow);// circle
		
			// 4th point check
			circle(debug_image, pointD, 3, red);
		
			imshow("ransac debug", debug_image);
			waitKey(0);
		}
		
		//check if the 4 point is on the circle
		if(abs(cv::norm(pointD - center) - radius) > radius_tolerance) continue;
				
		// vote
		std::vector<int> votes;
		std::vector<int> no_votes;
		for(int i = 0; i < (int)points.size(); i++) 
		{
			double vote_radius = norm(points[i] - center);
			
			if(abs(vote_radius - radius) < radius_tolerance) 
			{
				votes.push_back(i);
			}
			else
			{
				no_votes.push_back(i);
			}
		}
		
		// check votes vs circle_threshold
		if( (float)votes.size() / (2.0*CV_PI*radius) >= circle_threshold )
		{
			circles.push_back(Vec3f(x,y,radius));
			
			// voting debug image
			if(false)
			{
				Mat debug_image2 = edges.clone();
				cvtColor(debug_image2, debug_image2, CV_GRAY2RGB);
		
				Scalar yellow(0,255,255);
				Scalar green(0,255,0);
			
				circle(debug_image2, center, 3, yellow); // center
				circle(debug_image2, center, radius, yellow);// circle
			
				// draw points that voted
				for(int i = 0; i < (int)votes.size(); i++)
				{
					circle(debug_image2, points[votes[i]], 1, green);
				}
			
				imshow("ransac debug", debug_image2);
				waitKey(0);
			}
			
			// remove points from the set so they can't vote on multiple circles
			std::vector<Point2d> new_points;
			for(int i = 0; i < (int)no_votes.size(); i++)
			{
				new_points.push_back(points[no_votes[i]]);
			}
			points.clear();
			points = new_points;		
		}
		
		// stop RANSAC if there are few points left
		if((int)points.size() < points_threshold)
			break;
	}
	
	return;
}





