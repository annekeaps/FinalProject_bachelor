#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <time.h>
#include <stdio.h>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade, eyeR_cascade, eyeL_cascade, mouth_cascade;

void total(Mat& img, int& jml)
{
	jml = 0;
	for (int j = 0; j<img.rows; j++)
	{
		for (int i = 0; i<img.cols; i++)
		{
			jml = jml + img.at<uchar>(j, i);
		}
	}
	
}


void trackContours(Mat& img, size_t& z, Point& n)
{
	Mat dst, thres;
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;

	size_t coor_X = 0;
	size_t coor_Y = 0;
	int min_X1 = 0;
	size_t min_Y1 = 0;
	size_t max_X1 = 0;
	size_t max_Y1 = 0;
	size_t meanX = 0;
	size_t max_X = 0;
	int max_Y = 0;
	size_t min_X = 3333;
	size_t min_Y = 3333;

	cvtColor(img, dst, CV_BGR2GRAY);
	Mat img3;
	equalizeHist(dst, img3);

	//------------------------------------------------------------
	//preprocessing
	int jml2 = 0;
	total(img3, jml2);

	Mat img2 = img3.clone();
	
	std::sort(img2.data, img2.dataend); // call std::sort	
	cv::Mat image = img2.reshape(0, 1); // reshaped your WxH matrix into a 1x(W*H) vector	

	int thres2;
	thres2 = (0.1 * jml2) / 100;

	int x = (int)image.at<uchar>(0, (255 - thres2));

	//end preprocessing
	//-------------------------------------------------------------
	
	//threshold image to separate skin and eyebrow

	threshold(img3, thres, x, 255, 1);
	
	//find contours of the threshold image
	findContours(thres, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	//processing each point in contours
	for (size_t i = 0; i < contours.size(); i++)
	{
		for (size_t j = 0; j < contours[i].size(); j++)
		{
			coor_X = contours[i][j].x;
			coor_Y = contours[i][j].y;

			//left eyebrow and right eyebrow
			if ((z == 0) || (z == 1))
			{
				//find point which has a highest in Y-axis (lowest point in the image)
				if (coor_Y > max_Y)
				{
					max_Y = coor_Y;
					max_Y1 = coor_X;
				}

				//find point which has a lowest in Y-axis (highest point in the image)
				if (coor_Y < min_Y)
				{
					min_Y = coor_Y;
					min_Y1 = coor_X;
				}
			}

			//mouth
			else
			{
				//find point which has a highest in X-axis
				if (coor_X > max_X)
				{
					max_X = coor_X;
					max_X1 = coor_Y;
				}

				//find point which has a lowest in X-axis
				else if (coor_X < min_X)
				{
					min_X = coor_X;
					min_X1 = coor_Y;
				}

				//find point which has a lowest in Y-axis (highest point in the image)
				if (coor_Y < min_Y)
				{
					min_Y = coor_Y;
					min_Y1 = coor_X;
				}
			}
		}
	}

	//draw keypoints of the eyebrows
	if ((z == 0) || (z == 1))
	{
		line(img, Point(min_Y1, min_Y), Point(min_Y1, min_Y), Scalar(0, 0, 255), 4, 8, 0);
		line(img, Point(max_Y1, max_Y), Point(max_Y1, max_Y), Scalar(0, 0, 255), 4, 8, 0);

		//calculate a distance between highest and lowest keypoint
		n = Point((max_Y - min_Y), (0, 0));
	}

	//draw keypoints of the mouth
	else if (z == 2)
	{
		line(img, Point(min_X, min_X1), Point(min_X, min_X1), Scalar(0, 0, 255), 4, 8, 0);
		line(img, Point(min_Y1, min_Y), Point(min_Y1, min_Y), Scalar(0, 0, 255), 4, 8, 0);
		line(img, Point(max_X, max_X1), Point(max_X, max_X1), Scalar(0, 0, 255), 4, 8, 0);

		//X-coordinate for distance of the mouth corner (mouth's width)
		//Y-coordinate for distance of the top-mouth's height

		n = Point(min_X1, min_Y);
	}
}


void calculateExp(size_t& type, Point& neutral, Point& current, size_t& output)
{
	//0 for eyebrow
	if (type = 2)
	{
		//eyebrow keypoints location
		if (current.x > neutral.x + 2)
		{
			//0 for raised eyebrow
			output = 0;
		}
		else if (current.x < neutral.x)
		{
			//1 for eyebrow downs
			output = 1;
		}
		else
		{
			//2 for neutral
			output = 2;
		}
	}
	//1 for mouth
	else if (type = 1)
	{
		cout << neutral.x << "   " << current.x << endl;
		//mouth keypoints location		
		if (current.x > neutral.x)
		{
			//1 for smile
			output = 1;
		}
		else if (current.x < neutral.x)
		{
			//0 for pulling down
			output = 0;
		}
		else if ((current.x = neutral.x) && (current.y = neutral.y))
		{
			//2 for neutral
			output = 2;
		}
	}
}

void predictExp(size_t& type, size_t& category, size_t& type2, size_t& category2, size_t& result)
{
	//eyebrow raise means like expression
	if ((category2 == 0))
	{
		result = 0;

	}//eyebrow down means dislike expression
	else if ((category2 == 1))
	{
		if ((type == 1) && (category = 1))
		{
			result = 2;
		}
		else {
			result = 1;
		}
	}
	//lip pulling down means
	else if ((category2 == 2))
	{
		if (category = 1)
		{
			result = 0;
		}
		else if (category = 0)
		{
			result = 1;
		}
		else
		{
			result = 2;
		}

		//lip smile means like expression
	}
}

void calculateFoc(Point& minEB, Point& maxEB, Point& minCorner, Point& maxCorner, bool& output)
{
	if ((minEB.x = minCorner.x) || (maxEB.x = maxCorner.x))
	{
		output = false;
	}
	else
	{
		output = true;
	}
}


void eyePoint(Mat& img, Mat& src, Point& coor_min, Point& coor_max, bool& position)
{
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;

	//find contours of eyeball
	findContours(src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
	size_t max = 0;
	size_t index = 0;

	//find contour with has the biggest number of points
	for (size_t idx = 0; idx < contours.size(); idx++)
	{
		if (contours[idx].size() > max)
		{
			max = contours[idx].size();
			index = idx;
		}
	}

	size_t coor, min;
	size_t index_min = 0;
	size_t index_max = 0;

	//processing points in contour
	for (size_t i = 0; i < contours[index].size(); i++)
	{
		coor = contours[index][i].x;

		//initiate min and max value
		if (i == 0)
		{
			min = coor;
			max = 0;
		}

		//find the point with has lowest value in X-axis
		else if (coor < min)
		{
			min = coor;
			index_min = i;
			coor_min = Point(contours[index][index_min]);
		}

		//find the point with has highest value in X-axis
		else if (coor > max)
		{
			max = coor;
			index_max = i;
			coor_max = Point(contours[index][index_max]);
		}
	}

	//draw the highest value and lowest X-axis value point
	line(img, coor_min, coor_min, Scalar(0, 255, 0), 2, 8, 0);
	line(img, coor_max, coor_max, Scalar(0, 255, 0), 2, 8, 0);
}

void trackEyeBall(Mat& img, bool& result, bool& position)
{
	Mat satu, dua, gray, src;
	Point minEB(0, 0), min(0, 0), maxEB(0, 0), max(0, 0);
	cvtColor(img, gray, CV_BGR2GRAY, 1);
	imshow("aa", gray);

	//threshold for find the eyeball in the gray image

	//--------------------------------------------------------------------
	//preprocessing

	Mat img3;

	int jml2 = 0;
	total(gray, jml2);

	Mat img2 = gray.clone();	

	std::sort(img2.data, img2.dataend); // call std::sort	
	cv::Mat image = img2.reshape(0, 1); // reshaped your WxH matrix into a 1x(W*H) vector	

	double thres;
	thres = (0.02 * jml2) / 100;

	int x = (int)image.at<uchar>(0, thres);
	
	threshold(gray, src, x, 255, 1);

	imshow("eyeball", src);
	
	//eyeball tracking and draw keypoints
	eyePoint(img, src, minEB, maxEB, position);
	
	//threshold for find the eye corner in the gray image

	//-----------------------------------------------------------------
	//preprocessing eye corner

	jml2 = 0;
	total(gray, jml2);

	img2 = gray.clone();
	imshow("mata", img2);

	std::sort(img2.data, img2.dataend); // call std::sort
	image = img2.reshape(0, 1); // reshaped your WxH matrix into a 1x(W*H) vector

	double thres2;
	thres2 = (0.14 * jml2) / 100;

	x = (int)image.at<uchar>(0, (255 - thres2));
	
	threshold(gray, src, x, 255, 1);

	imshow("eye corner", src);

	//eye corner tracking and draw keypoints
	eyePoint(img, src, min, max, position);

	calculateFoc(minEB, maxEB, min, max, result);
}

void predictOp(size_t& exp, bool& focus, bool& result)
{
	if ((exp = 0) && (focus = true))
	{
		result = true;
	}
	else if ((exp = 1) && (focus = true))
	{
		result = false;
	}
	else if ((exp = 0) && (focus = false))
	{
		result = true;
	}
	else if ((exp = 1) && (focus = false))
	{
		result = false;
	}
	else if ((exp = 2) && (focus = true))
	{
		result = true;
	}
	else if ((exp = 2) && (focus = false))
	{
		result = false;
	}
}


int detectEye(size_t& z, Mat& im, Mat& tpl, Rect& rect)
{
	vector<Rect> faces, eyes, mouths;

	//find the face location
	face_cascade.detectMultiScale(im, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	size_t i, k;

	for (i = 0; i < faces.size(); i++)
	{
		Mat face = im(faces[i]);
		if (z != 2){

			//find the right eyebrow location
			if (z == 1)
			{
				eyeR_cascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

				if (eyes.size())
				{
					eyes[0].height = eyes[0].height + 5;
					eyes[0].width = eyes[0].width - 7;
					rect = eyes[0] + Point(faces[i].x - 3, faces[i].y - 9);
					tpl = im(rect);
				}
			}
			//find the left eyebrow location
			else if (z == 0)
			{
				eyeL_cascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

				if (eyes.size())
				{
					eyes[0].height = eyes[0].height + 5;
					eyes[0].width = eyes[0].width - 7;
					rect = eyes[0] + Point(faces[i].x + 5, faces[i].y - 9);
					tpl = im(rect);
				}
			}

			//find the right eye location
			if (z == 3)
			{
				eyeR_cascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

				if (eyes.size())
				{
					eyes[0].height = eyes[0].height - 8;
					eyes[0].width = eyes[0].width - 5;
					rect = eyes[0] + Point(faces[i].x, faces[i].y + 7);
					tpl = im(rect);
				}
			}

			//find the left eye location
			else if (z == 4)
			{
				eyeL_cascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));

				if (eyes.size())
				{
					eyes[0].height = eyes[0].height - 7;
					eyes[0].width = eyes[0].width - 5;
					rect = eyes[0] + Point(faces[i].x, faces[i].y + 7);
					tpl = im(rect);
				}
			}
		}

		//find the mouth location
		else if (z == 2){
			mouth_cascade.detectMultiScale(face, mouths);

			if (mouths.size())
			{
				for (k = 0; k < mouths.size(); k++)
				{
					mouths[0].width = mouths[0].width + 1;
					mouths[0].height = mouths[0].height - 1;
					rect = mouths[0] + Point(faces[i].x - 2, faces[i].y);
					tpl = im(rect);
				}
			}
		}
	}

	if (z == 2) {
		return mouths.size();
	}
	else {
		return eyes.size();
	}
}

void trackEye(Mat& im, Mat& tpl, Rect& rect)
{
	Size size(rect.width * 2, rect.height * 2);

	//create a new window with the same size
	Rect window(rect + size - Point(size.width / 2, size.height / 2));
	window &= Rect(0, 0, im.cols, im.rows);
	Mat dst(window.width - tpl.rows + 1, window.height - tpl.cols + 1, CV_32FC1);

	//matching the image in new window with template
	matchTemplate(im(window), tpl, dst, CV_TM_SQDIFF_NORMED);

	double minval, maxval;
	Point minloc, maxloc;

	//find minimum and maximum value and its location
	minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);

	if (minval <= 0.2)
	{
		rect.x = window.x + minloc.x;
		rect.y = window.y + minloc.y;
	}
	else
		rect.x = rect.y = rect.width = rect.height = 0;
}

int main(int argc, char** argv)
{
	int64 start, end;
	double sum = 0, exec;

	face_cascade.load("haarcascade_frontalface_alt2.xml");
	eyeL_cascade.load("haarcascade_mcs_lefteye.xml");
	eyeR_cascade.load("haarcascade_mcs_righteye.xml");
	mouth_cascade.load("haarcascade_mcs_mouth.xml");

	// Open webcam
	VideoCapture cap(0);

	// Check if everything is ok	
	if (face_cascade.empty() || eyeL_cascade.empty() || eyeR_cascade.empty() || mouth_cascade.empty() || !cap.isOpened())
		return 1;

	// Set video to 320x240
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	Mat frame, eyeR_tpl, eyeL_tpl, mouth_tpl, eyeER_tpl, eyeEL_tpl;
	Rect eyeR_bb, eyeL_bb, mouth_bb, mouth, eyeER_bb, eyeEL_bb;
	size_t zR = 1, zL = 0, zM = 2, eR = 3, eL = 4, loop = 0;
	Point distM(0, 0), distR(0, 0), distL(0, 0), distEL(0, 0), distER(0, 0);
	Point startM, startR, startL;
	int sum0 = 0, sum1 = 0, sum2 = 0;
	size_t sum0_2 = 0, sum1_2 = 0;
	size_t expResult = 2;
	bool resultEB_R = true, resultEB_L = true;
	bool output = true;

	while (1)
	{
		//execution time		
		start = getTickCount();
		cout << "--------------------------------------------------" << endl;
		cout << "\n\tstart: " << start << endl << endl;

		loop++;
		cap >> frame;
		if (frame.empty())
			break;

		// Flip the frame horizontally, Windows users might need this
		flip(frame, frame, 1);

		// Convert to grayscale		
		Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);
		imshow("start", frame);
		imshow("gray", gray);

		//check if eye and mouth are detected
		if ((eyeL_bb.width == 0 && eyeL_bb.height == 0) || (eyeR_bb.width == 0 && eyeR_bb.height == 0) || (mouth_bb.width == 0 && mouth_bb.height == 0))
		{
			//detect right and left eyes to be become the template
			detectEye(eR, gray, eyeER_tpl, eyeER_bb);
			detectEye(eL, gray, eyeEL_tpl, eyeEL_bb);

			//detect right and left eyebrows to be become the template
			detectEye(zR, gray, eyeR_tpl, eyeR_bb);
			detectEye(zL, gray, eyeL_tpl, eyeL_bb);

			//detect mouth to be become the template
			detectEye(zM, gray, mouth_tpl, mouth_bb);
		}
		else
		{
			// Tracking eyebrows with template matching
			trackEye(gray, eyeR_tpl, eyeR_bb);
			trackEye(gray, eyeL_tpl, eyeL_bb);

			// Tracking mouth with template matching
			trackEye(gray, mouth_tpl, mouth_bb);

			// Tracking eye with template matching
			trackEye(gray, eyeER_tpl, eyeER_bb);
			trackEye(gray, eyeEL_tpl, eyeEL_bb);

		}

		Point minEB_L(0, 0), min_L(0, 0), maxEB_L(0, 0), max_L(0, 0);
		Point minEB_R(0, 0), min_R(0, 0), maxEB_R(0, 0), max_R(0, 0);

		//select and draw keypoints of mouth
		trackContours(frame(mouth_bb), zM, distM);

		//select and draw keypoints of eyebrow
		trackContours(frame(eyeR_bb), zR, distR);
		trackContours(frame(eyeL_bb), zL, distL);

		bool rightEB;
		//select and draw keypoints of eyes
		rightEB = true;
		trackEyeBall(frame(eyeER_bb), resultEB_R, rightEB);
		rightEB = false;
		trackEyeBall(frame(eyeEL_bb), resultEB_L, rightEB);

		if ((resultEB_R = true) || (resultEB_L = true))
		{
		cout << "\t\tUser focused : focus " << endl;
		resultEB_R = true;
		}
		else
		{
		cout << "\t\tUser focused : unfocus " << endl;
		resultEB_R = false;
		}

		size_t expEye = 2, expM = 2;
		size_t typeEye = 2, typeM = 1;

		//decide the neutral points
		if (loop == 1)
		{
		startM = distM;
		startL = distL;
		startR = distR;
		}

		else if (loop > 1)
		{
		calculateExp(typeM, startM, distM, expM);
		calculateExp(typeEye, startL, distL, expEye);
		calculateExp(typeEye, startR, distR, expEye);
		predictExp(typeM, expM, typeEye, expEye, expResult);
		}

		if (expResult == 1)
		{
		cout << "\t\tFacial expression : dislike " << endl;
		}
		else if (expResult == 0)
		{
		cout << "\t\tFacial expression : like " << endl;
		}

		predictOp(expResult, resultEB_R, output);
		if (output = true)
		{
		cout << "\n\t\tPrediction of user opinion for this product: LIKE" << endl;
		sum0++;
		}
		else
		{
		cout << "\n\t\tPrediction of user opinion for this product: DISLIKE" << endl;
		sum1++;
		}

		end = getTickCount();
		cout << "\n\tend: " << end << endl;
		exec = (end - start) / getTickFrequency();
		cout << "\n\t\tExecution Time : " << exec << " seconds" << endl;
		sum = sum + exec;
		//		}

		// Display video
		imshow("video", frame);

		//press "esc" for exit
		if (waitKey(100) == 27)
		{
			break;
		}

	}

	cout << "\n\t===========================================================" << endl;
	cout << "\n\t\t\t\tSTOP" << endl;

	if (sum0 < sum1){
	cout << "\n\t\tThe Most User Opinion Prediction Result :  DISLIKE" << endl;
	}
	else
	{
	cout << "\n\t\tThe Most User Opinion Prediction Result :  LIKE" << endl;
	}

	cout << "\n\t\tAverage of execution time:  " << (sum / loop) << " seconds" << endl;

	waitKey(0);
	return 0;
}
