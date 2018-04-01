# reconstruction-with-two-images-
given two images to compute 3d point cloud just like stereo reconstruction

/**********************************************
* 本程序实现单目相机在已知轨迹下的三维重建
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

***

关键函数 **getPointCloud** 的参数详解

输入两张图像得到石头的点云及相关信息

* [input]: ref					第一张图像

* [input]: curr				第二张图像

* [input]: T_r_c				第二张相对于第一张的相机位姿

* [input]: intrinsics_name		相机内参文件名

* [output]:mid_stone			石头的质心（在世界坐标系中的三维坐标）

* [output]:stone_z				石头的高度（石头点云距离地面的最大距离）

* [output]:plane_coef			拟合的地面平面Ax+By+Cz+D=0的系数 , plane_coef[0] = A, plane_coef[1] = B, plane_coef[2] = C, plane_coef[3] = D;

* [output]:outlier_bbox		石头阴影在地面的外包围框

* [output]:inlier_bbox			石头的内包围框
