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

// for pcl
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <vtkActor.h> 
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL);

// for opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

// �����������
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

/**********************************************
* ������ʵ�ֵ�Ŀ�������֪�켣�µ���ά�ؽ�
*
*
*
* �������ŵ�Ŀͼ��:ref, curr
*
* ������֪pose������RT: T_r_w��ref_pose: world relative to reference camera��, T_c_w (curr_pose: world relative to current camera).
*
* ��������ڲ�:intrinsics.yml
*
*
*
*
*
��Ҫrectify��Ȼ����SGBM����Ӳ���ǻ��õ���ά���ꡣ
***********************************************/

//�ж�vector��ĳһԪ���Ƿ����  
bool is_element_in_vector(vector<int> v, int element) {
	vector<int>::iterator it;
	it = find(v.begin(), v.end(), element);
	if (it != v.end()) {
		return true;
	}
	else {
		return false;
	}
}

static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);
static cv::Mat toCvMat(const Eigen::Matrix3d &m);
static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);
static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);

void showpcd(PointCloud::Ptr &cloud);

cv::Mat RegionGrow(cv::Mat &in_img, cv::Point init_pos, int reg_maxdist, vector<cv::Point> &reg_choose);

//Ax+by+cz+D = 0  
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

void GetLine(cv::Point p1, cv::Point p2, float &a, float &b, float &c) {
	a = p2.y - p1.y;
	b = p1.x - p2.x;
	c = p2.x*p1.y - p1.x*p2.y;
}

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

float Point2Plane(PointT p, Eigen::VectorXf &coef, float norm_plane) {
	float dist = fabs(coef[0]*p.x + coef[1]*p.y + coef[2]*p.z + coef[3]) / norm_plane;
	return dist;
}

int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor);

int getPointCloud(cv::Mat &ref, cv::Mat &curr, Matrix4d &T_r_w, Matrix4d &T_c_w, const string &intrinsics_name, const string & dispname, const string  &pcd_name);
int getPointCloud(cv::Mat &ref, cv::Mat &curr, Matrix4d &T_r_c, const string &intrinsics_name, const string & dispname, const string  &pcd_name);

int main(int argc, char** argv)
{
	cv::Mat ref = cv::imread("G:\\Study\\codes\\reconstruction\\data\\first.tif", CV_LOAD_IMAGE_UNCHANGED);                // CV_LOAD_IMAGE_COLOR
	if (ref.empty()) {
		cout << "load reference image failed\n";
		system("pause");
		return -1;
	}
	//imshow("left", ref);
																														 //�ڶ���ͼ
	cv::Mat curr = cv::imread("G:\\Study\\codes\\reconstruction\\data\\last.tif", CV_LOAD_IMAGE_UNCHANGED);
	if (curr.empty()) {
		cout << "load current image failed\n";
		system("pause");
		return -1;
	}
	//imshow("right", curr);

	string intrinsics_name = "G:\\Study\\codes\\reconstruction\\data\\intrinsics.yml";
	string filename = "G:\\Study\\codes\\reconstruction\\data\\reproject_pcd.txt";
	string dispname = "G:\\Study\\codes\\reconstruction\\data\\sgbm.jpg";

	////�����

	//Matrix4d Tcw1; // reference Camera matrix:Rotation and Translation
	//Tcw1 << 0.04802580, -0.96432000, -0.26034800, 177.08100000, 0.99529700, 0.02424890, 0.09378320, 187.89500000, -0.08412380, -0.26362800, 0.96094900, 621.80700000,0,0,0,1;
	// 
	//Matrix4d Tcw2; // Current Camera matrix:Rotation and Translation
	//Tcw2 << 0.04802580, -0.96432000, -0.26034800, 77.10210000, 0.99529700, 0.02424890, 0.09378320, 186.45800000, -0.08412380, -0.26362800, 0.96094900, 620.24200000, 0, 0, 0, 1;
	//
	//getPointCloud(ref, curr, Tcw1, Tcw2, intrinsics_name, dispname, filename);

	//�����λ��
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

	cv::waitKey(0);
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

void showpcd(PointCloud::Ptr &cloud) {
	/*ͼ����ʾģ��*/
	//��ʾ����
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

	//���ñ���ɫ
	viewer->setBackgroundColor(0, 0, 0.7);

	//���õ�����ɫ���ô�Ϊ��һ��ɫ����
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);

	//�����Ҫ��ʾ�ĵ�������
	viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "sample cloud");

	//���õ���ʾ��С
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");


	//--------------------
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

cv::Mat RegionGrow(cv::Mat &in_img, cv::Point init_pos, int reg_maxdist, vector<cv::Point> &reg_choose) {
	int height = in_img.rows;
	int width = in_img.cols;
	cv::Mat out_img(height, width, CV_32F, cv::Scalar::all(0));
	// ������ʼ������Ϊ��ɫ
	float reg_mean = in_img.at<float>(init_pos.y, init_pos.x);
	out_img.at<float>(init_pos.y, init_pos.x) = 1;
	// �������������ĻҶ�ֵ���ܺ�
	float reg_sum = reg_mean;
	// �������������ĵ�����
	int reg_num = 1;// reg_choose.size()
	// ÿ���ж���Χ�˸����ַ��������ĸ���
	int count = 1;
	// ��¼��ѡ�������
	//vector<cv::Point> reg_choose;
	reg_choose.push_back(init_pos);
	// ��һ����
	int num = 1;
	while (count > 0)
	{
		// ��Χ�˸����з��������ĵ�ĻҶ�ֵ�ܺ�
		float s_temp = 0;
		count = 0;
		for (int k = 0; k < num; ++k) { // �������ĵ���б����������ظ�
			int row = reg_choose[reg_num - num + k].y;
			int col = reg_choose[reg_num - num + k].x;
			if (out_img.at<float>(row, col) == 1 && row > 0 && row < height && col > 0 && col < width) { // ��ȷ���Ҳ��Ǳ߽��ϵĵ�
				// 8����
				for (int r = -1; r <= 1; ++r)
					for (int c = -1; c <= 1; ++c) {
						// δ�������������������ĵ�
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

int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor)
{
	// ��ԭʼ�Ӳ����ݵ�λ��ת��Ϊ 8 λ  
	cv::Mat disp8u;
	if (disparity.depth() != CV_8U)
	{
		disparity.convertTo(disp8u, CV_8U, 255 / (64 * 16.));
	}
	else
	{
		disp8u = disparity;
	}


	// ת��Ϊα��ɫͼ�� �� �Ҷ�ͼ��  
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
	// �������ŵ�Ŀͼ��Ͷ�Ӧλ�ˣ��Լ�����ڲΣ��ó����Ʋ������ļ�
	if (ref.empty() || curr.empty()) {
		cout << "the iamge file is wrong\n";
		return 1;
	}
	//���ڲ�
	cv::FileStorage fs(intrinsics_name, CV_STORAGE_READ);

	cv::Mat  _M1, _D1, _M2, _D2;
	fs["M1"] >> _M1;
	fs["D1"] >> _D1;
	fs["M2"] >> _M2;
	fs["D2"] >> _D2;

	//����������λ�ˣ���Ŀ�������Ŀ���ڶ���ͼƬ��Ӧ���������ϵ����ڵ�һ��ͼ���Ӧ���������ϵ
	//T_c_r = T_c_w * T_w_r : reference camera relative to current camera
	//Matrix4d T_c_r = T_c_w * T_r_w.inverse();
	Matrix4d T_r_c = T_r_w * T_c_w.inverse();

	//�������λ�˽���ͼ��У��
	Matrix3d _R = T_r_c.block(0, 0, 3, 3);
	Vector3d _T = T_r_c.block(0, 3, 3, 1);
	cv::Mat R, T, R1, R2, P1, P2, Q, r, t;
	R = toCvMat(_R);
	T = toCvMat(_T);
	cout << "R = " << R << endl;
	cout << "T = " << T << endl;
	int rows_l = ref.rows;
	int cols_l = ref.cols;
	cv::Rect roi1, roi2;
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

	cv::Mat map11;
	cv::Mat map12;
	cv::Mat map21;
	cv::Mat map22;

	cv::initUndistortRectifyMap(_M1, _D1, R1, P1.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, map11, map12);
	cv::initUndistortRectifyMap(_M2, _D2, R2, P2.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, map21, map22);

	cv::Mat img1, left;
	cv::Mat img2, right;
	ref.copyTo(img1); curr.copyTo(img2);

	cv::Mat img(rows_l, cols_l * 2, CV_8UC3);//�߶�һ�������˫��
	//imshow("rectified", img);

	//У������
	remap(img1, left, map11, map12, cv::INTER_LINEAR);//��У��
	remap(img2, right, map21, map22, cv::INTER_LINEAR);//��У��

	cv::Mat imgPart1 = img(cv::Rect(0, 0, cols_l, rows_l));//ǳ����
	cv::Mat imgPart2 = img(cv::Rect(cols_l, 0, cols_l, rows_l));//ǳ����
	resize(left, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
	resize(right, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);

	//������
	for (int i = 0; i < img.rows; i += 32)
		line(img, cv::Point(0, i), cv::Point(img.cols, i), cv::Scalar(0, 255, 0), 1, 8);

	//��ʾ�ж�׼��ͼ��
	cv::Mat smallImg;//�����ҵķֱ���1:1��ʾ̫��������С��ʾ
	resize(img, smallImg, cv::Size(), 0.8, 0.8, CV_INTER_AREA);
	//namedWindow("rectified", WINDOW_NORMAL);
	imshow("rectified", smallImg);

	//sgbm �㷨
	cv::Mat disp;

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

	disp.convertTo(disp, CV_32F, 1.0 / 16);                //����16�õ���ʵ�Ӳ�ֵ  

	cv::Mat disp8U = cv::Mat(disp.rows, disp.cols, CV_8UC1);       //��ʾ  
	cv::Mat disp8U2 = cv::Mat(disp.rows, disp.cols, CV_8UC1);       //��ʾ  

	normalize(disp, disp8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	getDisparityImage(disp8U, disp8U2, 1);

	//      imshow("left", left);  
	//     imshow("right", right);  
	imshow("disparity", disp8U2);
	imwrite(dispname, disp8U2);

	//  Mat newMat;
	//         cv::reprojectImageTo3D( disp8U, newMat, Q,true,-1 );

	std::FILE* fp = std::fopen(pcd_name.c_str(), "wt");

	//�Լ���дreprojectImageTo3D
	cv::Mat_ <float> Q_ = Q;
	cout << "Q" << Q << endl;

	const double max_z = 10000;//��λ��ʲô
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
	// �������ŵ�Ŀͼ������λ�ˣ��Լ�����ڲΣ��ó����Ʋ������ļ�
	if (ref.empty() || curr.empty()) {
		cout << "the iamge file is wrong\n";
		return 1;
	}
	if (ref.channels() == 3 || curr.channels() == 3) {
		cvtColor(ref, ref, cv::COLOR_BGR2GRAY);
		cvtColor(curr, curr, cv::COLOR_BGR2GRAY);
	}
	//���ڲ�
	cv::FileStorage fs(intrinsics_name, CV_STORAGE_READ);

	cv::Mat  _M1, _D1, _M2, _D2;
	fs["M1"] >> _M1;
	fs["D1"] >> _D1;
	fs["M2"] >> _M2;
	fs["D2"] >> _D2;

	cout << "M1 = ";
	cout << _M1 << endl;

	//������λ�ˣ���Ŀ�������Ŀ���ڶ���ͼƬ��Ӧ���������ϵ����ڵ�һ��ͼ���Ӧ���������ϵ

	//�������λ�˽���ͼ��У��
	Matrix3d _R = T_r_c.block(0, 0, 3, 3);
	Vector3d _T = T_r_c.block(0, 3, 3, 1);
	cv::Mat R, T, R1, R2, P1, P2, Q, r, t;
	R = toCvMat(_R);
	T = toCvMat(_T);
	cout << "R = " << R << endl;
	cout << "T = " << T << endl;
	int rows_l = ref.rows;
	int cols_l = ref.cols;
	cout << "rows = " << rows_l << endl;
	cout << "cols = " << cols_l << endl;
	cv::Rect roi1, roi2;
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

	cv::Mat map11;
	cv::Mat map12;
	cv::Mat map21;
	cv::Mat map22;

	cv::initUndistortRectifyMap(_M1, _D1, R1, P1.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, map11, map12);
	cv::initUndistortRectifyMap(_M2, _D2, R2, P2.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F, map21, map22);

	int width = ref.cols;
	int height = ref.rows;

	cv::Mat img1, left(height, width, CV_8UC1);
	cv::Mat img2, right(height, width, CV_8UC1);
	ref.copyTo(img1); curr.copyTo(img2);

	//У������
	remap(img1, left, map11, map12, cv::INTER_LINEAR);//��У��
	remap(img2, right, map21, map22, cv::INTER_LINEAR);//��У��


	// allocate memory for disparity images
	const int32_t dims[3] = { width,height,width };
	cv::Mat disp_left(height, width, CV_32F);
	cv::Mat disp_right(height, width, CV_32F);

	// process
	Elas::parameters param;
	param.postprocess_only_left = false;
	Elas elas(param);
	elas.process(left.ptr<uchar>(0), right.ptr<uchar>(0), disp_left.ptr<float>(0), disp_right.ptr<float>(0), dims);

	imwrite("G:\\Study\\codes\\reconstruction\\data\\left.png",left);
	imwrite("G:\\Study\\codes\\reconstruction\\data\\right.png", right);


	// ��ʾ�ж�׼�Ķ���ͼ��
	//Mat img(rows_l, cols_l * 2, CV_8UC1);//�߶�һ�������˫��
	//Mat imgPart1 = img(Rect(0, 0, img.cols/2, img.rows));//ǳ����
	//Mat imgPart2 = img(Rect(img.cols / 2, 0, img.cols / 2, img.rows));//ǳ����
	//resize(left, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
	//resize(right, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);
	////������
	//for (int i = 0; i < img.rows; i += 32)
	//	line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1, 8);
	////��ʾ�ж�׼��ͼ��
	////Mat smallImg;//�����ҵķֱ���1:1��ʾ̫��������С��ʾ
	////resize(img, smallImg, Size(), 0.8, 0.8, CV_INTER_AREA);
	//imshow("rectified", img);


	cv::Mat disp8U = cv::Mat(height, width, CV_8UC1);       //��ʾ  
	cv::Mat disp8U2 = cv::Mat(height, width, CV_8UC1);       //��ʾ  

	normalize(disp_right, disp8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	getDisparityImage(disp8U, disp8U2, 1);

	cv::namedWindow("disparity", cv::WINDOW_NORMAL);
	cv::imshow("disparity", disp_left);
	cv::imwrite(dispname, disp8U2);
	cout << "disparity image saved\n";
	//cv::waitKey(3000);
	//string xyzw_name = "G:\\Study\\codes\\reconstruction\\data\\xyzw_pcd.txt";
	//string Q_name = "G:\\Study\\codes\\reconstruction\\data\\Q.txt";
	std::FILE* fp = std::fopen(pcd_name.c_str(), "wt");
	//std::FILE* xyzw = std::fopen(xyzw_name.c_str(), "wt");
	//std::FILE* Q_file = std::fopen(Q_name.c_str(), "wt");

	//�Լ���дreprojectImageTo3D
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
	const double max_z = 2.0;//��λ��ʲô,ȡ���ڴ����T
	PointCloud::Ptr cloud(new PointCloud);
	cv::Mat depth(height, width, CV_32F, cv::Scalar::all(0));
	float depth_max = 0;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			vec_tmp(0) = x;
			vec_tmp(1) = y;
			vec_tmp(2) = disp_right.at<float>(y, x);
			vec_tmp(3) = 1;
			cv::Point img_index;
			img_index.x = x;//��
			img_index.y = y;//��
			vec_tmp = Q_*vec_tmp;
			vec_tmp /= vec_tmp(3);
			PointT p;
			p.x = vec_tmp(0);
			p.y = vec_tmp(1);
			p.z = vec_tmp(2);
			if (p.z < 0.001 || fabs(p.z - max_z) < FLT_EPSILON || fabs(p.z) > max_z)  continue; 
			// �������
			cloud->points.push_back(p);
			img_indices.push_back(img_index);
			// ������ļ�
			fprintf(fp, "%f %f %f\n", p.x, p.y, p.z);
			//fprintf(xyzw, "%f %f %f %f\n", p.x, p.y, p.z, tmp_w);
			if (p.z > depth_max) depth_max = p.z;
			depth.at<float>(y, x) = p.z;
		}
	}
	fclose(fp);
	//fclose(xyzw);
	cout << "point cloud txt saved\n";

	depth = 255.0 * (depth / depth_max);
	imwrite("G:\\Study\\codes\\reconstruction\\data\\depth_right.png", depth);
	cout << "the max depth = " << depth_max << endl;
	//imshow("depth", depth);


	// ���ݵ������ƽ��
	clock_t start, finish;
	start = clock();


	//// ������ڵ�����
	//std::vector<int> inliers;
	//// ����һ����ģ�Ͷ���
	//pcl::SampleConsensusModelPlane<PointT>::Ptr model_p(new pcl::SampleConsensusModelPlane<PointT>(cloud));
	//pcl::RandomSampleConsensus<PointT> ransac(model_p);
	//ransac.setDistanceThreshold(0.01);
	//ransac.computeModel();
	//ransac.getInliers(inliers);

	//// ������������
	//vector<int> outliers;
	//for (int i = 0; i < cloud->points.size(); ++i)
	//	if (!is_element_in_vector(inliers, i))
	//		outliers.push_back(i);

	CvMat*points_mat = cvCreateMat(cloud->points.size(), 3, CV_32FC1);//���������洢��Ҫ��ϵ�ľ���   
	for (int i = 0; i < cloud->points.size(); ++i)
	{
		points_mat->data.fl[i * 3 + 0] = cloud->points[i].x;//�����ֵ���г�ʼ��   X������ֵ  
		points_mat->data.fl[i * 3 + 1] = cloud->points[i].y;//  Y������ֵ  
		points_mat->data.fl[i * 3 + 2] = cloud->points[i].z;//  Z������ֵ
	}
	float plane_coef[4] = { 0 };//������������ƽ�����������   
	cvFitPlane(points_mat, plane_coef);//���÷���,ƽ�淽��ΪAx+By+Cz+D = 0;
	Eigen::VectorXf coef = Eigen::VectorXf::Zero(4, 1);
	cout << "jym\n";
	for (int i = 0; i < 4; ++i)
	{
		coef[i] = plane_coef[i];
		cout << coef[i] << endl;
	}
	float norm_plane = coef.norm();
	float max_dist = 0;
	float min_dist = 0.02; // �㵽��ľ������2cm�ĵ���Ϊʯͷ�ϵĵ�
	int max_out_id = 0; // ʯͷ��ߵ������
	// ��������ϵĵ�����
	vector<int> inliers;
	// ����ʯͷ�ϵĵ�����
	vector<int> outliers;
	for (int i = 0; i < cloud->points.size(); ++i) {
		float dist = Point2Plane(cloud->points[i], coef, norm_plane);
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

	finish = clock();
	double time_used = double(finish - start) / CLOCKS_PER_SEC;
	cout << "ransac time used " << time_used << " s.\n";

	std::cout << "���ڵ㣺" << inliers.size() << std::endl;

	/*PointCloud::Ptr ground(new PointCloud);
	ground->resize(inliers.size());
	PointCloud::Ptr stone(new PointCloud);
	stone->resize(outliers.size());

	pcl::copyPointCloud(*cloud, inliers, *ground);
	pcl::copyPointCloud(*cloud, outliers, *stone);*/
	//pcl::io::savePCDFile("2.pcd", *final);



	//// ����ʯͷ���Ƶ������������
	//// ��ȡƽ�淽�̵Ĳ���
	//Eigen::VectorXf coef = Eigen::VectorXf::Zero(4, 1);
	//ransac.getModelCoefficients(coef);
	//float norm_plane = coef.norm();
	//int max_out_id = 0;
	//float max_dist = 0;
	//for (int i = 0; i < outliers.size(); ++i) {
	//	float dist = Point2Plane(cloud->points[outliers[i]], coef, norm_plane);
	//	if (dist > max_dist) {
	//		max_dist = dist;
	//		max_out_id = outliers[i];
	//	}
	//}	
	//cout << "jym\n";
	//for (int i = 0; i < 4; ++i)
	//{
	//	cout << coef[i] << endl;
	//}

	cout << "max distance from stone to ground is " << max_dist << endl;
	cout << "inliers size = " << inliers.size() << endl;
	cout << "outliers size = " << outliers.size() << endl;


	// �������ϵĵ���ԭͼ�϶�Ӧ�����ص��Ϊ0����ɫ������ʯͷ��Ӱ���������Ϊ1����ɫ��
	cv::Mat ground_img(height, width, CV_32F, cv::Scalar::all(1));
	for (int i = 0; i < inliers.size(); ++i)
		ground_img.at<float>(img_indices[inliers[i]].y, img_indices[inliers[i]].x) = 0;
	cv::Mat element5(5, 5, CV_8U, cv::Scalar(1));
	// ������
	morphologyEx(ground_img, ground_img, cv::MORPH_OPEN, element5);
	cv::imshow("ground img", ground_img);
	cv::waitKey(3000);


	// �������������ʯͷ���Χ��������
	// ��ʯͷ�е�һ�����ͶӰ����Ϊ���
	vector<cv::Point> stone_points;
	cv::Mat out_stone_img = RegionGrow(ground_img, img_indices[outliers[0]], 0, stone_points);
	//cv::imshow("stone_img", stone_img);
	// Ѱ����С��ת��Χ����
	cv::RotatedRect rectPoint = cv::minAreaRect(stone_points);
	// ����һ���洢�����ĸ��������ı���  
	cv::Point2f vertexPoint[4];
	// ��rectPoint�����д洢������ֵ�ŵ� fourPoint��������  
	rectPoint.points(vertexPoint);
	// ���ݵõ����ĸ��������  ���ƾ���  
	for (int i = 0; i < 3; ++i)
	{
		cv::line(out_stone_img, vertexPoint[i], vertexPoint[i + 1], cv::Scalar(1),3);
	}
	cv::line(out_stone_img, vertexPoint[0], vertexPoint[3], cv::Scalar(1), 3);

	cv::imshow("out bbox of stone", out_stone_img);

	// ���������ת���Σ�����������Ѱ��ʯͷ����С��Χ��

	// ���ʯͷ�ڲ�����
	// �������ص��Ϊ0����ɫ������ʯͷ�������������Ϊ1����ɫ��
	cv::Mat in_stone_img(height, width, CV_32F, cv::Scalar::all(0));
	for (int i = 0; i < outliers.size(); ++i) {
		in_stone_img.at<float>(img_indices[outliers[i]].y, img_indices[outliers[i]].x) = 1;
		out_stone_img.at<float>(img_indices[outliers[i]].y, img_indices[outliers[i]].x) = 0;
	}

	//cv::imshow("in_stone_img img", in_stone_img);
	//cv::Mat element5(5, 5, CV_8U, cv::Scalar(1));
	// ������
	morphologyEx(in_stone_img, in_stone_img, cv::MORPH_OPEN, element5);
	cv::imshow("in_stone_img img", in_stone_img);

	cv::Mat stone8U = cv::Mat(in_stone_img.rows, in_stone_img.cols, CV_8UC1);       //��ʾ  

	normalize(in_stone_img, stone8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	// ����ȡʯͷ����ͶӰ��ͼ��ƽ���γɵ��ڰ�Χ����
	vector <vector<cv::Point>>contours;
	cv::findContours(stone8U,
		contours,   //����������  
		CV_RETR_EXTERNAL,   //��ȡ������  
		CV_CHAIN_APPROX_NONE);  //��ȡÿ��������ÿ������
	// ѡȡ�����Ϊʯͷ�ڰ�Χ����
	int clen = 0;  
	int countour_id = 0;
	int index = 0;
	vector<cv::Point> m_contour;
	for (auto itc = contours.begin(); itc != contours.end(); ++itc) {
		if (itc->size() > clen) {
			clen = itc->size();
			m_contour = *itc;
			countour_id = index;
		}
		++index;
	}
	//�ڰ�ɫͼ���ϻ��ƺ�ɫ����  
	cv::Mat result_erase(stone8U.size(), CV_8U, cv::Scalar(255));
	drawContours(result_erase, contours,
		countour_id, //������������  
		cv::Scalar(0),  //��ɫΪ��ɫ  
		2); //�����ߵĻ��ƿ��Ϊ2  

	cv::namedWindow("contours_erase");
	cv::imshow("contours_erase", result_erase);

	// ���������ϵ�ÿ���㣬Ѱ�ҵ����������ߵ���̾����Ӧ�������㣬�Ӷ��õ��ڰ�Χ����
	// �ȿ�����ƽ�еĳ���
	float A01, B01, C01, A03, B03, C03;
	/*GetLine(vertexPoint[1], vertexPoint[2], A01, B01, C01);
	GetLine(vertexPoint[1], vertexPoint[0], A03, B03, C03);*/
	GetLine(vertexPoint[0], vertexPoint[1], A01, B01, C01);
	GetLine(vertexPoint[0], vertexPoint[3], A03, B03, C03);
	cv::line(result_erase, vertexPoint[1], vertexPoint[2], cv::Scalar(0), 3);
	cv::line(result_erase, vertexPoint[1], vertexPoint[0], cv::Scalar(0), 3);
	cout << "vertexPoint[0] = (" << vertexPoint[1].x << ", " << vertexPoint[1].y << ")\n";
	cout << "vertexPoint[1] = (" << vertexPoint[2].x << ", " << vertexPoint[2].y << ")\n";
	cout << "vertexPoint[3] = (" << vertexPoint[0].x << ", " << vertexPoint[0].y << ")\n";
	float norm01 = sqrtf(A01*A01 + B01*B01);
	float norm03 = sqrtf(A03*A03 + B03*B03);
	float min_dist01 = 9999, max_dist01 = 0;
	float min_dist03 = 9999, max_dist03 = 0;
	int min01, max01, min03, max03;// �߽�������
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
	/*cout << "min01 = " << min01 << endl;
	cout << "max01 = " << max01 << endl;
	cout << "min03 = " << min03 << endl;
	cout << "max03 = " << max03 << endl;
	cout << m_contour[min01] << endl;
	cout << m_contour[max01] << endl;
	cout << m_contour[min03] << endl;
	cout << m_contour[max03] << endl;*/
	//cv::circle(result_erase, m_contour[min01], 6, cv::Scalar(0));
	//cv::circle(result_erase, m_contour[max01], 7, cv::Scalar(0));
	//cv::circle(result_erase, m_contour[min03], 8, cv::Scalar(0));
	//cv::circle(result_erase, m_contour[max03], 9, cv::Scalar(0));
	// �ҵ��ĸ��߽����ҵ���������Χֱ�ߣ������󽻵㼴Ϊ�ڰ�Χ���ε��ĸ�����
	vector<cv::Point> inlier_vertexPoint;
	inlier_vertexPoint.push_back(Get2LinePoint(A01, B01, (-A01*m_contour[min01].x - B01*m_contour[min01].y), A03, B03, (-A03*m_contour[min03].x - B03*m_contour[min03].y)));
	inlier_vertexPoint.push_back(Get2LinePoint(A01, B01, (-A01*m_contour[min01].x - B01*m_contour[min01].y), A03, B03, (-A03*m_contour[max03].x - B03*m_contour[max03].y)));
	inlier_vertexPoint.push_back(Get2LinePoint(A01, B01, (-A01*m_contour[max01].x - B01*m_contour[max01].y), A03, B03, (-A03*m_contour[max03].x - B03*m_contour[max03].y)));
	inlier_vertexPoint.push_back(Get2LinePoint(A01, B01, (-A01*m_contour[max01].x - B01*m_contour[max01].y), A03, B03, (-A03*m_contour[min03].x - B03*m_contour[min03].y)));
	
	/*for (int i = 0; i < 4; ++i) {
		cv::circle(result_erase, inlier_vertexPoint[i], 16, cv::Scalar(0));
		cv::circle(result_erase, vertexPoint[i], i+10, cv::Scalar(0));
		cout << inlier_vertexPoint[i].x << endl;
		cout << inlier_vertexPoint[i].y << endl;
	}*/
	//cv::imshow("contours_erase", result_erase);
	// ��ʾ�ڰ�Χ��
	for (int i = 0; i < 3; ++i)
	{
		cv::line(out_stone_img, inlier_vertexPoint[i], inlier_vertexPoint[i + 1], cv::Scalar(0),3);
	}
	cv::line(out_stone_img, inlier_vertexPoint[0], inlier_vertexPoint[3], cv::Scalar(0), 3);

	cv::imshow("two bboxes of stone", out_stone_img);

	//// ��ʾ���ƣ�ʯͷ�͵���ֲ�ͬ��ɫ

	///*ͼ����ʾģ��*/
	////��ʾ����
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

	////���ñ���ɫ
	//viewer->setBackgroundColor(0, 0, 0);

	////���õ�����ɫ���ô�Ϊ��һ��ɫ����
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ground_color(ground, 0, 255, 0);
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> stone_color(stone, 0, 0, 255);

	////�����Ҫ��ʾ�ĵ�������
	//viewer->addPointCloud<pcl::PointXYZ>(ground, ground_color, "ground");
	//viewer->addPointCloud<pcl::PointXYZ>(stone, stone_color, "stone");

	////���õ���ʾ��С
	//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "ground");
	//viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "stone");


	////--------------------
	//while (!viewer->wasStopped())
	//{
	//	viewer->spinOnce(100);
	//	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	//}


	return 0;
}