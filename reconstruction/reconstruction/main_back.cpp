#include <iostream>
#include <vector>
#include <fstream>
using namespace std;
#include <boost/timer.hpp>
#include <boost/concept_check.hpp>
#include <cstdio>
#include <cstdlib>
// for eigen 
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

// for elas
#include "elas.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
using namespace cv;

/**********************************************
* 本程序实现单目相机在已知轨迹下的三维重建
*
*
*
* 传入两张单目图像:ref, curr
*
* 传入已知pose求出相对RT: T_r_w（ref_pose: world relative to reference camera）, T_c_w (curr_pose: world relative to current camera).
*
* 传入相机内参:intrinsics.yml
*
*
*
*
*
需要rectify，然后用SGBM求出视差，三角化得到三维坐标。
***********************************************/

static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);
static cv::Mat toCvMat(const Eigen::Matrix3d &m);
static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);
static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);

int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor);

int getPointCloud(cv::Mat &ref, cv::Mat &curr, Matrix4d &T_r_w, Matrix4d &T_c_w, const string &intrinsics_name, const string & dispname, const string  &pcd_name);
int getPointCloud(cv::Mat &ref, cv::Mat &curr, Matrix4d &T_r_c, const string &intrinsics_name, const string & dispname, const string  &pcd_name);

int main(int argc, char** argv)
{
	Mat ref = imread("G:\\Study\\codes\\reconstruction\\data\\first.tif", CV_LOAD_IMAGE_UNCHANGED);                // CV_LOAD_IMAGE_COLOR
	if (ref.empty()) {
		cout << "load reference image failed\n";
		system("pause");
		return -1;
	}
	//imshow("left", ref);
																														 //第二张图
	Mat curr = imread("G:\\Study\\codes\\reconstruction\\data\\last.tif", CV_LOAD_IMAGE_UNCHANGED);
	if (curr.empty()) {
		cout << "load current image failed\n";
		system("pause");
		return -1;
	}
	//imshow("right", curr);

	string intrinsics_name = "G:\\Study\\codes\\reconstruction\\data\\intrinsics.yml";
	string filename = "G:\\Study\\codes\\reconstruction\\data\\reproject_pcd.txt";
	string dispname = "G:\\Study\\codes\\reconstruction\\data\\sgbm.jpg";

	////读外参

	//Matrix4d Tcw1; // reference Camera matrix:Rotation and Translation
	//Tcw1 << 0.04802580, -0.96432000, -0.26034800, 177.08100000, 0.99529700, 0.02424890, 0.09378320, 187.89500000, -0.08412380, -0.26362800, 0.96094900, 621.80700000,0,0,0,1;
	// 
	//Matrix4d Tcw2; // Current Camera matrix:Rotation and Translation
	//Tcw2 << 0.04802580, -0.96432000, -0.26034800, 77.10210000, 0.99529700, 0.02424890, 0.09378320, 186.45800000, -0.08412380, -0.26362800, 0.96094900, 620.24200000, 0, 0, 0, 1;
	//
	//getPointCloud(ref, curr, Tcw1, Tcw2, intrinsics_name, dispname, filename);

	//读相对位姿
	Matrix4d T_r_c;
	/*T_r_c << 1, 0, 0, 99.7847,
		0, 1, 0, 20.9048,
		0, 0, 1, 2.45157,
		0, 0, 0, 1;*/
	T_r_c << 1   , 0 ,0, 0.0997847,
	0 ,   1  ,  0, 0.0209048,
	0 ,0 ,   1, 0.00245157,
	0, 0, 0, 1;

	/*T_r_c << 0.9774, 0.1068, -0.1822, 0.0491619,
		-0.1211, 0.9902, -0.0691, -0.0197851,
		0.1731, 0.0896, 0.9808, 0.0314907,
		0, 0, 0, 1;*/
	    
		     
		        

	/*T_r_c << 0.9904   , 0.1216 ,- 0.0662, -0.0119192,
		-0.1078 ,   0.9772  ,  0.1829, -0.0785181,
		0.0869 ,- 0.1740 ,   0.9809, 0.0375545,
		0, 0, 0, 1;*/
	/*T_r_c << 0.9906, 0.1173, 0.0705, 0.021397,
		-0.1027, 0.9775, -0.1841, 0.0274153,
		-0.0905, 0.1752, 0.9804, 0.0201863,
		0, 0, 0, 1;*/
		
	getPointCloud(ref, curr, T_r_c, intrinsics_name, dispname, filename);

	waitKey(0);
	system("pause");
	return 0;
}


cv::Mat toCvMat(const Eigen::Matrix3d &m)
{
	cv::Mat cvMat(3, 3, CV_32F);
	for (int i = 0; i<3; i++)
		for (int j = 0; j<3; j++)
			cvMat.at<float>(i, j) = m(i, j);

	return cvMat.clone();
}

cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m)
{
	cv::Mat cvMat(3, 1, CV_32F);
	for (int i = 0; i<3; i++)
		cvMat.at<float>(i) = m(i);

	return cvMat.clone();
}
cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t) {
	cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			cvMat.at<float>(i, j) = R(i, j);
		}
	}
	for (int i = 0; i < 3; i++) {
		cvMat.at<float>(i, 3) = t(i);
	}

	return cvMat.clone();
}

int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor)
{
	// 将原始视差数据的位深转换为 8 位  
	cv::Mat disp8u;
	if (disparity.depth() != CV_8U)
	{
		disparity.convertTo(disp8u, CV_8U, 255 / (64 * 16.));
	}
	else
	{
		disp8u = disparity;
	}


	// 转换为伪彩色图像 或 灰度图像  
	if (isColor)
	{
		if (disparityImage.empty() || disparityImage.type() != CV_8UC3)
		{
			disparityImage = cv::Mat::zeros(disparity.rows, disparity.cols, CV_8UC3);
		}


		for (int y = 0; y<disparity.rows; y++)
		{
			for (int x = 0; x<disparity.cols; x++)
			{
				uchar val = disp8u.at<uchar>(y, x);
				uchar r, g, b;


				if (val == 0)
					r = g = b = 0;
				else
				{
					r = 255 - val;
					g = val < 128 ? val * 2 : (uchar)((255 - val) * 2);
					b = val;
				}
				disparityImage.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);
			}
		}
	}
	else
	{
		disp8u.copyTo(disparityImage);
	}

	return 1;
}

int getPointCloud(cv::Mat &ref, cv::Mat &curr, Matrix4d &T_r_w, Matrix4d &T_c_w, const string &intrinsics_name, const string & dispname, const string &pcd_name) {
	// 传入两张单目图像和对应位姿，以及相机内参，得出点云并存入文件
	if (ref.empty() || curr.empty()) {
		cout << "the iamge file is wrong\n";
		return 1;
	}
	//读内参
	FileStorage fs(intrinsics_name, CV_STORAGE_READ);

	Mat  _M1, _D1, _M2, _D2;
	fs["M1"] >> _M1;
	fs["D1"] >> _D1;
	fs["M2"] >> _M2;
	fs["D2"] >> _D2;

	//计算相机相对位姿，右目相对于左目，第二张图片对应的相机坐标系相对于第一张图像对应的相机坐标系
	//T_c_r = T_c_w * T_w_r : reference camera relative to current camera
	//Matrix4d T_c_r = T_c_w * T_r_w.inverse();
	Matrix4d T_r_c = T_r_w * T_c_w.inverse();

	//利用相对位姿进行图像校正
	Matrix3d _R = T_r_c.block(0, 0, 3, 3);
	Vector3d _T = T_r_c.block(0, 3, 3, 1);
	Mat R, T, R1, R2, P1, P2, Q, r, t;
	R = toCvMat(_R);
	T = toCvMat(_T);
	cout << "R = " << R << endl;
	cout << "T = " << T << endl;
	int rows_l = ref.rows;
	int cols_l = ref.cols;
	Rect roi1, roi2;
	R.convertTo(r, CV_64F);
	T.convertTo(t, CV_64F);

	cout << "begin stereo rectify\n";
	//
	cv::stereoRectify(_M1, _D1, _M2, _D2, cv::Size(cols_l, rows_l), r, -t, R1, R2, P1, P2, Q);
	// 	  cv::stereoRectify( _M1, _D1,_M2, _D2, cv::Size(cols_l,rows_l), r, t, R1, R2, P1, P2, Q,CALIB_ZERO_DISPARITY, 1,  cv::Size(cols_l,rows_l), &roi1, &roi2);
	cout << "R1\n" << R1 << endl;
	cout << "R2\n" << R2 << endl;
	cout << "P1\n" << P1 << endl;
	cout << "P2\n" << P2 << endl;

	Mat map11;
	Mat map12;
	Mat map21;
	Mat map22;

	cv::initUndistortRectifyMap(_M1, _D1, R1, P1.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, map11, map12);
	cv::initUndistortRectifyMap(_M2, _D2, R2, P2.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, map21, map22);

	Mat img1, left;
	Mat img2, right;
	ref.copyTo(img1); curr.copyTo(img2);

	Mat img(rows_l, cols_l * 2, CV_8UC3);//高度一样，宽度双倍
	//imshow("rectified", img);

	//校正对齐
	remap(img1, left, map11, map12, cv::INTER_LINEAR);//左校正
	remap(img2, right, map21, map22, cv::INTER_LINEAR);//右校正

	Mat imgPart1 = img(Rect(0, 0, cols_l, rows_l));//浅拷贝
	Mat imgPart2 = img(Rect(cols_l, 0, cols_l, rows_l));//浅拷贝
	resize(left, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
	resize(right, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);

	//画横线
	for (int i = 0; i < img.rows; i += 32)
		line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1, 8);

	//显示行对准的图形
	Mat smallImg;//由于我的分辨率1:1显示太大，所以缩小显示
	resize(img, smallImg, Size(), 0.8, 0.8, CV_INTER_AREA);
	//namedWindow("rectified", WINDOW_NORMAL);
	imshow("rectified", smallImg);

	//sgbm 算法
	Mat disp;

	int mindisparity = 0;
	//     int ndisparities = 64;  
	int ndisparities = 144;

	//     int SADWindowSize = 11;   
	int SADWindowSize = 9;

	//     SGBM  
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
	int P11 = 8 * left.channels() * SADWindowSize* SADWindowSize;
	//         int P11 = 4 * left.channels() * SADWindowSize* SADWindowSize;  

	int P22 = 32 * left.channels() * SADWindowSize* SADWindowSize;
	sgbm->setP1(P11);
	sgbm->setP2(P22);

	sgbm->setPreFilterCap(15);
	//         sgbm->setPreFilterCap(30);  

	sgbm->setUniquenessRatio(2);
	//         sgbm->setUniquenessRatio(15);  

	sgbm->setSpeckleRange(2);

	//     sgbm->setSpeckleWindowSize(100);  
	sgbm->setSpeckleWindowSize(10);

	sgbm->setDisp12MaxDiff(1);

	//     sgbm->setMode(cv::StereoSGBM::MODE_HH);  
	sgbm->compute(left, right, disp);

	disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值  

	Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示  
	Mat disp8U2 = Mat(disp.rows, disp.cols, CV_8UC1);       //显示  

	normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
	getDisparityImage(disp8U, disp8U2, 1);

	//      imshow("left", left);  
	//     imshow("right", right);  
	imshow("disparity", disp8U2);
	imwrite(dispname, disp8U2);

	//  Mat newMat;
	//         cv::reprojectImageTo3D( disp8U, newMat, Q,true,-1 );

	std::FILE* fp = std::fopen(pcd_name.c_str(), "wt");

	//自己重写reprojectImageTo3D
	cv::Mat_ <float> Q_ = Q;
	cout << "Q" << Q << endl;

	const double max_z = 10000;//单位是什么
	cv::Mat_<cv::Vec3f> XYZ(disp.rows, disp.cols);   // Output point cloud
	cv::Mat_<float> vec_tmp(4, 1);
	for (int y = 0; y < disp.rows; ++y) {
		for (int x = 0; x < disp.cols; ++x) {
			vec_tmp(0) = x;
			vec_tmp(1) = y;
			vec_tmp(2) = disp.at<float>(y, x);
			vec_tmp(3) = 1;
			vec_tmp = Q_*vec_tmp;
			vec_tmp /= vec_tmp(3);
			cv::Vec3f &point = XYZ.at<cv::Vec3f>(y, x);
			point[0] = vec_tmp(0);
			point[1] = vec_tmp(1);
			point[2] = vec_tmp(2);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
	return 0;
}

int getPointCloud(cv::Mat &ref, cv::Mat &curr, Matrix4d &T_r_c, const string &intrinsics_name, const string & dispname, const string &pcd_name) {
	// 传入两张单目图像和相对位姿，以及相机内参，得出点云并存入文件
	if (ref.empty() || curr.empty()) {
		cout << "the iamge file is wrong\n";
		return 1;
	}
	if (ref.channels() == 3 || curr.channels() == 3) {
		cvtColor(ref, ref, COLOR_BGR2GRAY);
		cvtColor(curr, curr, COLOR_BGR2GRAY);
	}
	//读内参
	FileStorage fs(intrinsics_name, CV_STORAGE_READ);

	Mat  _M1, _D1, _M2, _D2;
	fs["M1"] >> _M1;
	fs["D1"] >> _D1;
	fs["M2"] >> _M2;
	fs["D2"] >> _D2;

	cout << "M1 = ";
	cout << _M1 << endl;

	//相机相对位姿，右目相对于左目，第二张图片对应的相机坐标系相对于第一张图像对应的相机坐标系

	//利用相对位姿进行图像校正
	Matrix3d _R = T_r_c.block(0, 0, 3, 3);
	Vector3d _T = T_r_c.block(0, 3, 3, 1);
	Mat R, T, R1, R2, P1, P2, Q, r, t;
	R = toCvMat(_R);
	T = toCvMat(_T);
	cout << "R = " << R << endl;
	cout << "T = " << T << endl;
	int rows_l = ref.rows;
	int cols_l = ref.cols;
	cout << "rows = " << rows_l << endl;
	cout << "cols = " << cols_l << endl;
	Rect roi1, roi2;
	R.convertTo(r, CV_64F);
	T.convertTo(t, CV_64F);

	cout << "begin stereo rectify\n";
	//
	cv::stereoRectify(_M1, _D1, _M2, _D2, cv::Size(cols_l, rows_l), r, t, R1, R2, P1, P2, Q);
	// 	  cv::stereoRectify( _M1, _D1,_M2, _D2, cv::Size(cols_l,rows_l), r, t, R1, R2, P1, P2, Q,CALIB_ZERO_DISPARITY, 1,  cv::Size(cols_l,rows_l), &roi1, &roi2);
	cout << "R1" << R1 << endl;
	cout << "R2" << R2 << endl;
	cout << "P1" << P1 << endl;
	cout << "P2" << P2 << endl;

	Mat map11;
	Mat map12;
	Mat map21;
	Mat map22;

	cv::initUndistortRectifyMap(_M1, _D1, R1, P1.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, map11, map12);
	cv::initUndistortRectifyMap(_M2, _D2, R2, P2.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, map21, map22);

	int width = ref.cols;
	int height = ref.rows;

	Mat img1, left(height, width, CV_8UC1);
	Mat img2, right(height, width, CV_8UC1);
	ref.copyTo(img1); curr.copyTo(img2);

	
	//Mat img(rows_l, cols_l * 2, CV_8UC1);//高度一样，宽度双倍
										 //imshow("rectified", img);

										 //校正对齐
	remap(img1, left, map11, map12, cv::INTER_LINEAR);//左校正
	remap(img2, right, map21, map22, cv::INTER_LINEAR);//右校正


	// allocate memory for disparity images
	const int32_t dims[3] = { width,height,width }; // bytes per line = width
	//float* D1_data = (float*)malloc(width*height * sizeof(float));
	//float* D2_data = (float*)malloc(width*height * sizeof(float));
	Mat disp_left(height, width, CV_32F);
	Mat disp_right(height, width, CV_32F);

	// process
	Elas::parameters param;
	param.postprocess_only_left = false;
	Elas elas(param);
	elas.process(left.ptr<uchar>(0), right.ptr<uchar>(0), disp_left.ptr<float>(0), disp_right.ptr<float>(0), dims);
	//elas.process(left.ptr<uchar>(0), right.ptr<uchar>(0), D1_data, D2_data, dims);

	imwrite("G:\\Study\\codes\\reconstruction\\data\\left.jpg",left);
	imwrite("G:\\Study\\codes\\reconstruction\\data\\right.jpg", right);

	//Mat imgPart1 = img(Rect(0, 0, img.cols/2, img.rows));//浅拷贝
	//Mat imgPart2 = img(Rect(img.cols / 2, 0, img.cols / 2, img.rows));//浅拷贝
	//resize(left, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
	//resize(right, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);

	////cout << "img size = " << img.size() << endl;
	////namedWindow("rectified", WINDOW_NORMAL);
	////imshow("rectified", img);

	////画横线
	//for (int i = 0; i < img.rows; i += 32)
	//	line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1, 8);

	////显示行对准的图形
	////Mat smallImg;//由于我的分辨率1:1显示太大，所以缩小显示
	////resize(img, smallImg, Size(), 0.8, 0.8, CV_INTER_AREA);
	//imshow("rectified", img);



	////sgbm 算法
	//Mat disp;

	//int mindisparity = 0;
	////int ndisparities = 64;  
	//int ndisparities = 144;

	//int SADWindowSize = 11;   
	////int SADWindowSize = 11;

	////     SGBM  
	//cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
	//int P11 = 8 * left.channels() * SADWindowSize* SADWindowSize;
	////         int P11 = 4 * left.channels() * SADWindowSize* SADWindowSize;  

	//int P22 = 32 * left.channels() * SADWindowSize* SADWindowSize;
	//sgbm->setP1(P11);
	//sgbm->setP2(P22);

	//sgbm->setPreFilterCap(15);//15
	////         sgbm->setPreFilterCap(15);  

	//sgbm->setUniquenessRatio(15);//15
	////         sgbm->setUniquenessRatio(15);  

	//sgbm->setSpeckleRange(2);//2

	////     sgbm->setSpeckleWindowSize(100);  
	//sgbm->setSpeckleWindowSize(10);//10

	//sgbm->setDisp12MaxDiff(1);

	////     sgbm->setMode(cv::StereoSGBM::MODE_HH);  
	//sgbm->compute(left, right, disp);

	//disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值  

	Mat disp8U = Mat(disp_right.rows, disp_right.cols, CV_8UC1);       //显示  
	Mat disp8U2 = Mat(disp_right.rows, disp_right.cols, CV_8UC1);       //显示  

	normalize(disp_right, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
	getDisparityImage(disp8U, disp8U2, 1);

	//////      imshow("left", left);  
	//////     imshow("right", right);  
	namedWindow("disparity", WINDOW_NORMAL);
	imshow("disparity", disp8U2);
	//imwrite(dispname, disp8U2);
	//cout << "disparity image saved\n";

	/*Mat depth;
	cv::reprojectImageTo3D( disp, depth, Q,true,-1 );
	Mat depth8U = Mat(depth.rows, depth.cols, CV_8UC1);
	normalize(depth, depth8U, 0, 255, NORM_MINMAX, CV_8UC1);
	imwrite("G:\\Study\\codes\\reconstruction\\data\\depth.jpg", depth);
	imshow("depth", depth8U);*/

	string xyzw_name = "G:\\Study\\codes\\reconstruction\\data\\xyzw_pcd.txt";
	string Q_name = "G:\\Study\\codes\\reconstruction\\data\\Q.txt";
	std::FILE* fp = std::fopen(pcd_name.c_str(), "wt");
	std::FILE* xyzw = std::fopen(xyzw_name.c_str(), "wt");
	std::FILE* Q_file = std::fopen(Q_name.c_str(), "wt");

	//自己重写reprojectImageTo3D
	//Q.at<float>(3, 2) = -Q.at<float>(3, 2);
	cv::Mat_ <float> Q_ = Q;
	Q_.at<float>(3, 2) = -Q_.at<float>(3, 2);
	cout << "Q" << Q_ << endl;
	fprintf(Q_file, "%f %f %f %f\n", Q_.at<float>(0,3), Q_.at<float>(1, 3), Q_.at<float>(2, 3), Q_.at<float>(3, 2));
	fclose(Q_file);

	// open disp.txt
	//char *name = "F:\\安装包\\libelas\\img\\stone_right_disp.txt";
	//ifstream disp_file(name);

	const double max_z = 2.0;//单位是什么,取决于传入的T
	double point_w = 0;
	cv::Mat_<cv::Vec3f> XYZ(disp_left.rows, disp_left.cols);   // Output point cloud
	cv::Mat_<float> vec_tmp(4, 1);
	cv::Mat depth(disp_left.rows, disp_left.cols, CV_32F, cv::Scalar::all(0));
	float depth_max = 0;
	for (int y = 0; y < disp_right.rows; ++y) {
		for (int x = 0; x < disp_right.cols; ++x) {
			vec_tmp(0) = x;
			vec_tmp(1) = y;
			vec_tmp(2) = disp_right.at<float>(y, x);
			//disp_file >> vec_tmp(2);
			//vec_tmp(2) = D2_data[x + y * width];
			vec_tmp(3) = 1;
			vec_tmp = Q_*vec_tmp;
			point_w = vec_tmp(3);
			vec_tmp /= vec_tmp(3);
			cv::Vec3f &point = XYZ.at<cv::Vec3f>(y, x);
			point[0] = vec_tmp(0);
			point[1] = vec_tmp(1);
			point[2] = vec_tmp(2);
			if (point[2] < 0.2) continue;
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
			fprintf(xyzw, "%f %f %f %f\n", point[0], point[1], point[2], point_w);
			if (point[2] > depth_max) depth_max = point[2];
			depth.at<float>(y, x) = point[2];
		}
	}
	fclose(fp);
	fclose(xyzw);
	cout << "point cloud txt saved\n";
	//disp_file.close();

	depth = 255.0 * (depth / depth_max);
	imwrite("G:\\Study\\codes\\reconstruction\\data\\depth.jpg", depth);
	cout << "the max depth = " << depth_max << endl;
	imshow("depth", depth);

	return 0;
}