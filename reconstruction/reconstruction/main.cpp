#include <iostream>
#include <vector>
#include <fstream>
using namespace std;
#include <boost/timer.hpp>
#include <boost/concept_check.hpp>
#include <cstdio>
#include <cstdlib>
#include <time.h>

// for eigen 
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

// for elas
#include "elas.h"

// for opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


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
* 需要rectify，然后用elas求出视差，三角化得到三维坐标
*
*
* 输出为石头的高度，石头在地面投影的内外包围框
***********************************************/

string first_name = "../data/first.tif";
string last_name = "../data/last.tif";
string intrinsics_name = "../data/intrinsics.yml";
string pcd_name = "../data/reproject_pcd.txt";
string disp_name = "../data/sgbm.png";
string rect_left = "../data/left.png";
string rect_right = "../data/right.png";
string depth_left_name = "../data/depth_left.png";


cv::Mat toCvMat(const Eigen::Matrix3d &m)
{
	cv::Mat cvMat(3, 3, CV_32F);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			cvMat.at<float>(i, j) = m(i, j);

	return cvMat.clone();
}
cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m)
{
	cv::Mat cvMat(3, 1, CV_32F);
	for (int i = 0; i < 3; i++)
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

// 区域增长
cv::Mat RegionGrow(cv::Mat &in_img, cv::Point init_pos, int reg_maxdist, vector<cv::Point> &reg_choose) {
	int height = in_img.rows;
	int width = in_img.cols;
	cv::Mat out_img(height, width, CV_32F, cv::Scalar::all(0));
	// 生长起始点设置为白色
	float reg_mean = in_img.at<float>(init_pos.y, init_pos.x);
	out_img.at<float>(init_pos.y, init_pos.x) = 1;
	// 符合生长条件的灰度值的总和
	float reg_sum = reg_mean;
	// 符合生长条件的点总数
	int reg_num = 1;// reg_choose.size()
					// 每次判断周围八个点种符合条件的个数
	int count = 1;
	// 记录已选点的坐标
	//vector<cv::Point> reg_choose;
	reg_choose.push_back(init_pos);
	// 第一个点
	int num = 1;
	while (count > 0)
	{
		// 周围八个点中符合条件的点的灰度值总和
		float s_temp = 0;
		count = 0;
		for (int k = 0; k < num; ++k) { // 对新增的点进行遍历，避免重复
			int row = reg_choose[reg_num - num + k].y;
			int col = reg_choose[reg_num - num + k].x;
			if (out_img.at<float>(row, col) == 1 && row > 0 && row < height && col > 0 && col < width) { // 已确定且不是边界上的点
																										 // 8邻域
				for (int r = -1; r <= 1; ++r)
					for (int c = -1; c <= 1; ++c) {
						// 未处理且满足生长条件的点
						if (out_img.at<float>(row + r, col + c) == 0 && fabs(in_img.at<float>(row + r, col + c) - reg_mean) <= reg_maxdist) {
							out_img.at<float>(row + r, col + c) = 1;
							count += 1;
							cv::Point p;
							p.x = col + c;
							p.y = row + r;
							reg_choose.push_back(p);
							s_temp += in_img.at<float>(row + r, col + c);
						}
					}
			}
		}
		num = count;
		reg_num += count;
		reg_sum += s_temp;
		reg_mean = reg_sum / reg_num;
	}
	return out_img;
}

// 拟合平面：Ax+by+cz+D = 0  
void cvFitPlane(const CvMat* points, float* plane) {
	// Estimate geometric centroid.  
	int nrows = points->rows;
	int ncols = points->cols;
	int type = points->type;
	CvMat* centroid = cvCreateMat(1, ncols, type);
	cvSet(centroid, cvScalar(0));
	for (int c = 0; c < ncols; c++) {
		for (int r = 0; r < nrows; r++)
		{
			centroid->data.fl[c] += points->data.fl[ncols*r + c];
		}
		centroid->data.fl[c] /= nrows;
	}
	// Subtract geometric centroid from each point.  
	CvMat* points2 = cvCreateMat(nrows, ncols, type);
	for (int r = 0; r < nrows; r++)
		for (int c = 0; c < ncols; c++)
			points2->data.fl[ncols*r + c] = points->data.fl[ncols*r + c] - centroid->data.fl[c];
	// Evaluate SVD of covariance matrix.  
	CvMat* A = cvCreateMat(ncols, ncols, type);
	CvMat* W = cvCreateMat(ncols, ncols, type);
	CvMat* V = cvCreateMat(ncols, ncols, type);
	cvGEMM(points2, points, 1, NULL, 0, A, CV_GEMM_A_T);
	cvSVD(A, W, NULL, V, CV_SVD_V_T);
	// Assign plane coefficients by singular vector corresponding to smallest singular value.  
	plane[ncols] = 0;
	for (int c = 0; c < ncols; c++) {
		plane[c] = V->data.fl[ncols*(ncols - 1) + c];
		plane[ncols] += plane[c] * centroid->data.fl[c];
	}
	plane[3] = -plane[3];
	// Release allocated resources.  
	cvReleaseMat(&centroid);
	cvReleaseMat(&points2);
	cvReleaseMat(&A);
	cvReleaseMat(&W);
	cvReleaseMat(&V);
}

// 两点求直线
void GetLine(cv::Point p1, cv::Point p2, float &a, float &b, float &c) {
	a = p2.y - p1.y;
	b = p1.x - p2.x;
	c = p2.x*p1.y - p1.x*p2.y;
}

// 两条直线求交点
cv::Point Get2LinePoint(float a1, float b1, float c1, float a2, float b2, float c2) {
	cv::Point p;
	float x = (b1*c2 - b2*c1) / (a1*b2 - a2*b1);
	float y = (a2*c1 - a1*c2) / (a1*b2 - a2*b1);
	p.x = int(x);
	p.y = int(y);
	/*cout << "x = " << p.x << endl;
	cout << "y = " << p.y << endl;*/
	return p;
}

// 返回点到平面的距离
float Point2Plane(cv::Point3f p, Eigen::VectorXf &coef, float norm_plane) {
	// 提前计算出norm，防止大量重复计算
	float dist = fabs(coef[0]*p.x + coef[1]*p.y + coef[2]*p.z + coef[3]) / norm_plane;
	return dist;
}

// 返回点在平面的投影点
cv::Point3f ReprojectPoint2Plane(cv::Point3f p, Eigen::VectorXf &coef) {
	// 将Ax+by+cz+D=0 转化为 z = b1 + b2*x + b3*y
	float b1 = -coef[3] / coef[2];
	float b2 = -coef[0] / coef[2];
	float b3 = -coef[1] / coef[2];
	cv::Point3f re_p;
	re_p.z = (b1 + b2*b2*p.z + b2*p.x + b3*b3*p.z + b3*p.y) / (1 + b2*b2 + b3*b3);
	re_p.x = (p.z - re_p.z)*b2 + p.x;
	re_p.y = (p.z - re_p.z)*b3 + p.y;
	return re_p;
}

// 得到伪彩色视差图
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

// 输入两张图像得到石头的点云及相关信息
// [input]: ref					第一张图像
// [input]: curr				第二张图像
// [input]: T_r_c				第二张相对于第一张的相机位姿
// [input]: intrinsics_name		相机内参文件名
// [output]:mid_stone			石头的质心（在世界坐标系中的三维坐标）
// [output]:stone_z				石头的高度（石头点云距离地面的最大距离）
// [output]:plane_coef			拟合的地面平面Ax+By+Cz+D=0的系数 , plane_coef[0] = A, plane_coef[1] = B, plane_coef[2] = C, plane_coef[3] = D;
// [output]:outlier_bbox		石头阴影在地面的外包围框
// [output]:inlier_bbox			石头的内包围框
int getPointCloud(cv::Mat &ref, cv::Mat &curr, Matrix4d &T_r_c, const string &intrinsics_name, cv::Point3f &mid_stone, float &stone_z, float *plane_coef, vector<cv::Point2f> &outlier_bbox, vector<cv::Point> &inlier_bbox) {
	// 传入两张单目图像和相对位姿，以及相机内参，得出点云并存入文件
	if (ref.empty() || curr.empty()) {
		cout << "the iamge file is wrong\n";
		return 1;
	}
	if (ref.channels() == 3 || curr.channels() == 3) {
		cvtColor(ref, ref, cv::COLOR_BGR2GRAY);
		cvtColor(curr, curr, cv::COLOR_BGR2GRAY);
	}
	//读内参
	cv::FileStorage fs(intrinsics_name, CV_STORAGE_READ);

	cv::Mat  _M1, _D1, _M2, _D2;
	fs["M1"] >> _M1;
	fs["D1"] >> _D1;
	fs["M2"] >> _M2;
	fs["D2"] >> _D2;

	/*cout << "M1 = ";
	cout << _M1 << endl;*/

	//相机相对位姿，右目相对于左目，第二张图片对应的相机坐标系相对于第一张图像对应的相机坐标系

	//利用相对位姿进行图像校正
	Matrix3d _R = T_r_c.block(0, 0, 3, 3);
	Vector3d _T = T_r_c.block(0, 3, 3, 1);
	cv::Mat R, T, R1, R2, P1, P2, Q, r, t;
	R = toCvMat(_R);
	T = toCvMat(_T);
	/*cout << "R = " << R << endl;
	cout << "T = " << T << endl;*/
	int width = ref.cols;
	int height = ref.rows;
	cv::Rect roi1, roi2;
	R.convertTo(r, CV_64F);
	T.convertTo(t, CV_64F);

	cout << "begin stereo rectify\n";
	//
	cv::stereoRectify(_M1, _D1, _M2, _D2, cv::Size(width, height), r, t, R1, R2, P1, P2, Q);
	//cv::stereoRectify( _M1, _D1,_M2, _D2, cv::Size(width,height), r, t, R1, R2, P1, P2, Q,CALIB_ZERO_DISPARITY, 1,  cv::Size(width,height), &roi1, &roi2);
	cout << "R1\n" << R1 << endl;
	cout << "R2\n" << R2 << endl;
	cout << "P1\n" << P1 << endl;
	cout << "P2\n" << P2 << endl;

	cv::Mat map11;
	cv::Mat map12;
	cv::Mat map21;
	cv::Mat map22;

	cv::initUndistortRectifyMap(_M1, _D1, R1, P1.rowRange(0, 3).colRange(0, 3), cv::Size(width, height), CV_32F, map11, map12);
	cv::initUndistortRectifyMap(_M2, _D2, R2, P2.rowRange(0, 3).colRange(0, 3), cv::Size(width, height), CV_32F, map21, map22);

	cv::Mat img1, left(height, width, CV_8UC1);
	cv::Mat img2, right(height, width, CV_8UC1);
	ref.copyTo(img1); curr.copyTo(img2);

	//校正对齐
	remap(img1, left, map11, map12, cv::INTER_LINEAR);//左校正
	remap(img2, right, map21, map22, cv::INTER_LINEAR);//右校正

	imwrite(rect_left, left);
	imwrite(rect_right, right);

	// elas方法计算视差
	// allocate memory for disparity images
	const int32_t dims[3] = { width,height,width };
	cv::Mat disp_left(height, width, CV_32F);
	cv::Mat disp_right(height, width, CV_32F);

	// process
	Elas::parameters param;
	param.postprocess_only_left = false;
	Elas elas(param);
	elas.process(left.ptr<uchar>(0), right.ptr<uchar>(0), disp_left.ptr<float>(0), disp_right.ptr<float>(0), dims);


	// 显示行对准的对齐图像
	//Mat img(height, width * 2, CV_8UC1);//高度一样，宽度双倍
	//Mat imgPart1 = img(Rect(0, 0, img.cols/2, img.rows));//浅拷贝
	//Mat imgPart2 = img(Rect(img.cols / 2, 0, img.cols / 2, img.rows));//浅拷贝
	//resize(left, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
	//resize(right, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);
	////画横线
	//for (int i = 0; i < img.rows; i += 32)
	//	line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1, 8);
	////显示行对准的图形
	////Mat smallImg;//由于我的分辨率1:1显示太大，所以缩小显示
	////resize(img, smallImg, Size(), 0.8, 0.8, CV_INTER_AREA);
	//imshow("rectified", img);


	cv::Mat disp8U = cv::Mat(height, width, CV_8UC1);       //显示  
	cv::Mat disp8U2 = cv::Mat(height, width, CV_8UC1);       //显示  

	normalize(disp_right, disp8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	getDisparityImage(disp8U, disp8U2, 1);

	cv::namedWindow("disparity", cv::WINDOW_NORMAL);
	cv::imshow("disparity", disp8U2);
	cv::imwrite(disp_name, disp8U2);
	cout << "disparity image saved\n";


	// 根据视差算三维点云
	std::FILE* fp = std::fopen(pcd_name.c_str(), "wt");

	//自己重写reprojectImageTo3D
	cv::Mat_ <float> Q_ = Q;
	Q_.at<float>(3, 2) = -Q_.at<float>(3, 2);
	cout << "Q" << Q_ << endl;
	/*Matrix4f Q_final;
	for (int i = 0; i < 4; ++i)
	for (int j = 0; j < 4; ++j)
	Q_final(i, j) = Q_.at<float>(i, j);*/

	//Vector4f vec_tmp;
	cv::Mat_<float> vec_tmp(4, 1);
	vector<cv::Point> img_indices;
	vector<cv::Point3f> cloud;
	const double max_z = 3.0;//单位是什么,取决于传入的T
	cv::Mat depth(height, width, CV_32F, cv::Scalar::all(0));
	float depth_max = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			vec_tmp(0) = x;
			vec_tmp(1) = y;
			vec_tmp(2) = disp_left.at<float>(y, x);
			vec_tmp(3) = 1;
			cv::Point img_index;
			img_index.x = x;//列
			img_index.y = y;//行
			vec_tmp = Q_*vec_tmp;
			vec_tmp /= vec_tmp(3);
			cv::Point3f p;
			p.x = vec_tmp(0);
			p.y = vec_tmp(1);
			p.z = vec_tmp(2);
			if (p.z < 0.001 || fabs(p.z - max_z) < FLT_EPSILON || fabs(p.z) > max_z)  continue;
			// 保存点云
			cloud.push_back(p);
			img_indices.push_back(img_index);
			// 输出至文件
			fprintf(fp, "%f %f %f\n", p.x, p.y, p.z);
			//fprintf(xyzw, "%f %f %f %f\n", p.x, p.y, p.z, tmp_w);
			if (p.z > depth_max) depth_max = p.z;
			depth.at<float>(y, x) = p.z;
		}
	}
	fclose(fp);
	cout << "point cloud txt saved\n";

	depth = 255.0 * (depth / depth_max);
	imwrite(depth_left_name, depth);
	//cout << "the max depth = " << depth_max << endl;
	//imshow("depth", depth);

	


	// 根据点云拟合平面
	CvMat*points_mat = cvCreateMat(cloud.size(), 3, CV_32FC1);//定义用来存储需要拟合点的矩阵   
	for (int i = 0; i < cloud.size(); ++i)
	{
		points_mat->data.fl[i * 3 + 0] = cloud[i].x;//矩阵的值进行初始化   X的坐标值  
		points_mat->data.fl[i * 3 + 1] = cloud[i].y;//  Y的坐标值  
		points_mat->data.fl[i * 3 + 2] = cloud[i].z;//  Z的坐标值
	}
	//plane_coef:定义用来储存平面参数的数组   
	cvFitPlane(points_mat, plane_coef);//调用方程,平面方程为Ax+By+Cz+D = 0;
	Eigen::VectorXf coef = Eigen::VectorXf::Zero(4, 1);
	for (int i = 0; i < 4; ++i)
	{
		coef[i] = plane_coef[i];
		//cout << coef[i] << endl;
	}
	float norm_plane = coef.norm();
	float max_dist = 0;
	float min_dist = 0.02; // 点到面的距离大于2cm的点视为石头上的点
	int max_out_id = 0; // 石头最高点的索引
	// 保存地面上的点索引
	vector<int> inliers;
	// 保存石头上的点索引
	vector<int> outliers;
	for (int i = 0; i < cloud.size(); ++i) {
		float dist = Point2Plane(cloud[i], coef, norm_plane);
		if (dist > max_dist) {
			max_dist = dist;
			max_out_id = i;
		}
		if (dist > min_dist) {
			outliers.push_back(i);
		}
		else {
			inliers.push_back(i);
		}
	}
	stone_z = max_dist;
	cout << "max distance from stone to ground is " << max_dist  << " m." << endl;
	cout << "inliers size = " << inliers.size() << endl;
	cout << "outliers size = " << outliers.size() << endl;

	// 根据石头上的点云和拟合的地面方程求石头的质心
	float maxx = 0, maxy = 0;
	float minx = 9999, miny = 9999;
	for (int i = 0; i < outliers.size(); ++i) {
		if (cloud[outliers[i]].x > maxx) maxx = cloud[outliers[i]].x;
		if (cloud[outliers[i]].y > maxy) maxy = cloud[outliers[i]].y;
		if (cloud[outliers[i]].x < minx) minx = cloud[outliers[i]].x;
		if (cloud[outliers[i]].y < miny) miny = cloud[outliers[i]].y;
	}
	cv::Point3f midp_high;
	midp_high.x = (maxx + minx) / 2;
	midp_high.y = (maxy + miny) / 2;
	midp_high.z = cloud[max_out_id].z; // 距离地面最高的点的z值
	// 计算点(midx, midy, maxz)到地面的投影点
	cv::Point3f midp_low;
	midp_low = ReprojectPoint2Plane(midp_high, coef);
	// 质心为这两点的中点
	mid_stone = (midp_high + midp_low) / 2;
	cout << "the mid point of stone is " << mid_stone << endl;

	// 将地面上的点在原图上对应的像素点标为0（黑色），将石头阴影所在区域标为1（白色）
	cv::Mat ground_img(height, width, CV_32F, cv::Scalar::all(1));
	for (int i = 0; i < inliers.size(); ++i)
		ground_img.at<float>(img_indices[inliers[i]].y, img_indices[inliers[i]].x) = 0;
	cv::Mat element5(5, 5, CV_8U, cv::Scalar(1));
	// 开运算
	morphologyEx(ground_img, ground_img, cv::MORPH_OPEN, element5);
	//cv::imshow("ground img", ground_img);




	// 区域增长，获得石头外包围完整区域
	// 将石头中的一个点的投影点作为起点
	vector<cv::Point> stone_points;
	cv::Mat out_stone_img = RegionGrow(ground_img, img_indices[outliers[0]], 0, stone_points);
	//cv::imshow("stone_img", stone_img);
	// 寻找最小旋转包围矩形
	cv::RotatedRect rectPoint = cv::minAreaRect(stone_points);
	// 定义一个存储以上四个点的坐标的变量  
	cv::Point2f vertexPoint[4];
	// 将rectPoint变量中存储的坐标值放到 fourPoint的数组中  
	rectPoint.points(vertexPoint);
	// 根据得到的四个点的坐标  绘制矩形  
	for (int i = 0; i < 3; ++i)
		cv::line(out_stone_img, vertexPoint[i], vertexPoint[i + 1], cv::Scalar(1), 3);
	cv::line(out_stone_img, vertexPoint[0], vertexPoint[3], cv::Scalar(1), 3);
	//cv::imshow("out bbox of stone", out_stone_img);
	for (int i = 0; i < 4; ++i)
		outlier_bbox.push_back(vertexPoint[i]);




	// 根据这个旋转矩形，向内收缩，寻找石头的最小包围框

	// 获得石头内部区域
	// 背景像素点标为0（黑色），将石头点云所在区域标为1（白色）
	cv::Mat in_stone_img(height, width, CV_32F, cv::Scalar::all(0));
	for (int i = 0; i < outliers.size(); ++i) {
		in_stone_img.at<float>(img_indices[outliers[i]].y, img_indices[outliers[i]].x) = 1;
		out_stone_img.at<float>(img_indices[outliers[i]].y, img_indices[outliers[i]].x) = 0;
	}
	//cv::imshow("in_stone_img img", in_stone_img);
	// 开运算
	morphologyEx(in_stone_img, in_stone_img, cv::MORPH_OPEN, element5);
	//cv::imshow("in_stone_img img", in_stone_img);
	cv::Mat stone8U = cv::Mat(in_stone_img.rows, in_stone_img.cols, CV_8UC1); //用于寻找轮廓
	normalize(in_stone_img, stone8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	// 先提取石头点云投影到图像平面形成的内包围轮廓
	vector <vector<cv::Point>>contours;
	cv::findContours(stone8U,
		contours,   //轮廓的数组  
		CV_RETR_EXTERNAL,   //获取外轮廓  
		CV_CHAIN_APPROX_NONE);  //获取每个轮廓的每个像素
	// 选取最长轮廓为石头内包围轮廓
	int clen = 0;
	//int countour_id = 0;
	int index = 0;
	vector<cv::Point> m_contour;
	for (auto itc = contours.begin(); itc != contours.end(); ++itc) {
		if (itc->size() > clen) {
			clen = itc->size();
			m_contour = *itc;
			//countour_id = index;
		}
		++index;
	}
	////在白色图像上绘制黑色轮廓  
	//cv::Mat result_erase(stone8U.size(), CV_8U, cv::Scalar(255));
	//drawContours(result_erase, contours,
	//	countour_id, //绘制所有轮廓  
	//	cv::Scalar(0),  //颜色为黑色  
	//	2); //轮廓线的绘制宽度为2  

	//cv::namedWindow("contours_erase");
	//cv::imshow("contours_erase", result_erase);



	// 遍历轮廓上的每个点，寻找到矩形四条边的最短距离对应的轮廓点，从而得到内包围矩形
	// 先看两个平行的长边
	float A01, B01, C01, A03, B03, C03;
	GetLine(vertexPoint[0], vertexPoint[1], A01, B01, C01);
	GetLine(vertexPoint[0], vertexPoint[3], A03, B03, C03);
	float norm01 = sqrtf(A01*A01 + B01*B01);
	float norm03 = sqrtf(A03*A03 + B03*B03);
	float min_dist01 = 9999, max_dist01 = 0;
	float min_dist03 = 9999, max_dist03 = 0;
	int min01, max01, min03, max03;// 边界点的索引
	for (int i = 0; i < m_contour.size(); ++i) {
		float dist01 = fabs((A01*m_contour[i].x + B01*m_contour[i].y + C01) / norm01);
		if (dist01 < min_dist01) {
			min_dist01 = dist01;
			min01 = i;
		}
		if (dist01 > max_dist01) {
			max_dist01 = dist01;
			max01 = i;
		}
		float dist03 = fabs((A03*m_contour[i].x + B03*m_contour[i].y + C03) / norm03);
		if (dist03 < min_dist03) {
			min_dist03 = dist03;
			min03 = i;
		}
		if (dist03 > max_dist03) {
			max_dist03 = dist03;
			max03 = i;
		}
	}

	// 找到四个边界点就找到了四条包围直线，两两求交点即为内包围矩形的四个顶点
	inlier_bbox.push_back(Get2LinePoint(A01, B01, (-A01*m_contour[min01].x - B01*m_contour[min01].y), A03, B03, (-A03*m_contour[min03].x - B03*m_contour[min03].y)));
	inlier_bbox.push_back(Get2LinePoint(A01, B01, (-A01*m_contour[min01].x - B01*m_contour[min01].y), A03, B03, (-A03*m_contour[max03].x - B03*m_contour[max03].y)));
	inlier_bbox.push_back(Get2LinePoint(A01, B01, (-A01*m_contour[max01].x - B01*m_contour[max01].y), A03, B03, (-A03*m_contour[max03].x - B03*m_contour[max03].y)));
	inlier_bbox.push_back(Get2LinePoint(A01, B01, (-A01*m_contour[max01].x - B01*m_contour[max01].y), A03, B03, (-A03*m_contour[min03].x - B03*m_contour[min03].y)));


	// 显示内包围框
	for (int i = 0; i < 3; ++i)
	{
		cv::line(out_stone_img, inlier_bbox[i], inlier_bbox[i + 1], cv::Scalar(0), 3);
	}
	cv::line(out_stone_img, inlier_bbox[0], inlier_bbox[3], cv::Scalar(0), 3);

	cv::imshow("two bboxes of stone", out_stone_img);


	return 0;
}

int main(int argc, char** argv)
{
	cv::Mat ref = cv::imread(first_name, CV_LOAD_IMAGE_UNCHANGED);        // CV_LOAD_IMAGE_COLOR
	if (ref.empty()) {
		cout << "load reference image failed\n";
		system("pause");
		return -1;
	}
	//imshow("left", ref);
	cv::Mat curr = cv::imread(last_name, CV_LOAD_IMAGE_UNCHANGED);        //第二张图
	if (curr.empty()) {
		cout << "load current image failed\n";
		system("pause");
		return -1;
	}
	//imshow("right", curr);


	//读相对位姿
	Matrix4d T_r_c;
	/*T_r_c << 1, 0, 0, 99.7847,
		0, 1, 0, 20.9048,
		0, 0, 1, 2.45157,
		0, 0, 0, 1;*/
	T_r_c << 1, 0, 0, 0.0997847,
			 0, 1, 0, 0.0209048,
			 0, 0, 1, 0.00245157,
			 0, 0, 0, 1;

	clock_t start, finish;// 计时
	start = clock();

	// 储存拟合平面的参数：Ax+By+Cz+D=0 , plane_coef[0] = A, plane_coef[1] = B, plane_coef[2] = C, plane_coef[3] = D;
	float plane_coef[4] = { 0 };
	float stone_height = 0;
	cv::Point3f mid_stone;
	vector<cv::Point2f> outlier_bbox;
	vector<cv::Point> inlier_bbox;

	getPointCloud(ref, curr, T_r_c, intrinsics_name, mid_stone, stone_height, plane_coef, outlier_bbox, inlier_bbox);

	finish = clock();
	double time_used = double(finish - start) / CLOCKS_PER_SEC;
	cout << "time used " << time_used << " s.\n";

	cv::waitKey(0);
	system("pause");
	return 0;
}


