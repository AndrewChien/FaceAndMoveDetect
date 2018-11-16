///运动物体检测——帧差法
#include "opencv2/opencv.hpp"
using namespace cv;
#include <iostream>
using namespace std;
//运动物体检测函数声明
Mat MoveDetect(Mat temp, Mat frame);

int main()
{

	VideoCapture video("bike.avi");//定义VideoCapture类video
	if (!video.isOpened())	//对video进行异常检测
	{
		cout << "video open error!" << endl;
		return 0;
	}
	int frameCount = video.get(CV_CAP_PROP_FRAME_COUNT);//获取帧数
	double FPS = video.get(CV_CAP_PROP_FPS);//获取FPS
	Mat frame;//存储帧
	Mat temp;//存储前一帧图像
	Mat result;//存储结果图像
	for (int i = 0; i < frameCount; i++)
	{

		video >> frame;//读帧进frame
		imshow("frame", frame);
		if (frame.empty())//对帧进行异常检测
		{
			cout << "frame is empty!" << endl;
			break;
		}
		if (i == 0)//如果为第一帧（temp还为空）
		{
			result = MoveDetect(frame, frame);//调用MoveDetect()进行运动物体检测，返回值存入result
		}
		else//若不是第一帧（temp有值了）
		{
			result = MoveDetect(temp, frame);//调用MoveDetect()进行运动物体检测，返回值存入result

		}
		imshow("result", result);
		if (waitKey(1000.0 / FPS) == 27)//按原FPS显示
		{
			cout << "ESC退出!" << endl;
			break;
		}
		temp = frame.clone();
	}
	return 0;


}
Mat MoveDetect(Mat temp, Mat frame)
{
	Mat result = frame.clone();
	//1.将background和frame转为灰度图
	Mat gray1, gray2;
	cvtColor(temp, gray1, CV_BGR2GRAY);
	cvtColor(frame, gray2, CV_BGR2GRAY);
	//2.将background和frame做差
	Mat diff;
	absdiff(gray1, gray2, diff);
	imshow("diff", diff);
	//3.对差值图diff_thresh进行阈值化处理
	Mat diff_thresh;
	threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);
	imshow("diff_thresh", diff_thresh);
	//4.腐蚀
	Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(18, 18));
	erode(diff_thresh, diff_thresh, kernel_erode);
	imshow("erode", diff_thresh);
	//5.膨胀
	dilate(diff_thresh, diff_thresh, kernel_dilate);
	imshow("dilate", diff_thresh);
	//6.查找轮廓并绘制轮廓
	vector<vector<Point>> contours;
	findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓
															 //7.查找正外接矩形
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//在result上绘制正外接矩形
	}
	return result;//返回result
}