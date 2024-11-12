#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <Eigen\Dense>
#include <string>
#include <io.h>
#include <fstream>



using namespace Eigen;
using namespace std;
using namespace cv;

Mat R_T2HomogeneousMatrix(const Mat& R, const Mat& T);
void HomogeneousMtr2RT(Mat& HomoMtr, Mat& R, Mat& T);
bool isRotatedMatrix(Mat& R);
Mat eulerAngleToRotateMatrix(const Mat& eulerAngle, const std::string& seq);
Mat quaternionToRotatedMatrix(const Vec4d& q);
Mat attitudeVectorToMatrix(const Mat& m, bool useQuaternion, const string& seq);
void m_calibration(vector<string>& FilesName, Size board_size, Size square_size, Mat& cameraMatrix, Mat& distCoeffs, vector<Mat>& rvecsMat, vector<Mat>& tvecsMat, vector<Mat>& rvecsMat0, vector<Mat>& tvecsMat0);



//数据使用的已有的值
//相机中13组标定板的位姿，x,y,z，rx,ry,rz,
/*
Mat_<double> CalPose = (cv::Mat_<double>(13, 6) <<
	-0.072944147641399, -0.06687830562048944, 0.4340418493881254, -0.2207496117519063, 0.0256862005614321, 0.1926014162476009,
	-0.01969337858360518, -0.05095294728651902, 0.3671266719105768, 0.1552099329677287, -0.5763323472739464, 0.09956130526058841,
	0.1358164536530692, -0.1110802522656379, 0.4001396735998251, -0.04486168331242635, -0.004268942058870162, 0.05290073845562016,
	0.1360676260120161, -0.002373036366121294, 0.3951670952829301, -0.4359637938379769, 0.00807193982932386, 0.06162504121755787,
	-0.1047666520352697, -0.01377729010376614, 0.4570029374109721, -0.612072103513551, -0.04939465180949879, -0.1075464055169537,
	0.02866460103085085, -0.1043911269729344, 0.3879127305077527, 0.3137563103168434, -0.02113958397023016, 0.1311397970432597,
	0.1122741829392126, 0.001044006395747612, 0.3686697279333643, 0.1607160803445018, 0.2468677059920437, 0.1035103912091547,
	-0.06079521129779342, -0.02815190820828123, 0.4451740202390909, 0.1280935541917056, -0.2674407142401368, 0.1633865613363686,
	-0.02475533256363622, -0.06950841248698086, 0.2939836207787282, 0.1260629671933584, -0.2637748974005461, 0.1634102148863728,
	0.1128618887222624, 0.00117877722121125, 0.3362496409334229, 0.1049541359309871, -0.2754352318773509, 0.4251492928748009,
	0.1510545750008333, -0.0725019944548204, 0.3369908269102371, 0.2615745097093249, -0.1295598776133405, 0.6974394284203849,
	0.04885313290076512, -0.06488755216394324, 0.2441532410787161, 0.1998243391807502, -0.04919417529483511, -0.05133193756053007,
	0.08816140480523708, -0.05549965109057759, 0.3164905645998022, 0.164693654482863, 0.1153894876338608, 0.01455551646362294);

/*
Mat_<double> ToolPose = (cv::Mat_<double>(16, 6) <<
-0.49231285,  0.18113587,  0.24438224, -2.58457221, -1.76926073, -0.03413765,
-0.44666499,  0.20994852,  0.24436463, -2.58456951, -1.76929599, -0.03433315,
-0.48189749,  0.12119823,  0.24439126, -2.58449784, -1.7691939, -0.03420369,
-0.56628937,  0.12674834,  0.24438187, -2.58448124, -1.76937623, -0.03423968,
-0.53009673,  0.24997356,  0.24440751, -2.58448124, -1.76933143, -0.0343062,
-0.53130604,  0.18499097,  0.22159947, -2.33897148, -2.08740848, -0.03583381,
-0.48691669,  0.18499775,  0.22157271, -2.8293713, -1.33562957, -0.03149124,
-0.52492348,  0.11494154,  0.22158581, -2.82935514, -1.33586435, -0.03139239,
-0.52567897,  0.15355645,  0.22158586, -2.82949388, -1.33560952, -0.03150532,
-0.5373281,  0.24869937,  0.26473846, -2.35653314, -2.0673832, -0.03566578,
-0.53729278,  0.14117137,  0.26472849, -2.35676239, -2.06719262, -0.03590201,
-0.48011845,  0.14113973,  0.26476989, -2.3566664, -2.06716078, -0.03566359,
-0.47718025,  0.13960469,  0.26730678, -2.91012513, -1.17270361, -0.04329864,
-0.54959481,  0.13962604,  0.2673394, -2.96013399, -1.03596063, -0.04288144,
-0.4587878,   0.18526934,  0.26734815, -2.95994396, -1.03606047, -0.04293956,
-0.49721658,  0.18193714,  0.26989715,  2.87110418,  1.00354908, -0.1663514); //04标定数据
*/
//机械臂末端13组位姿,x,y,z,rx,ry,rz
/*
Mat_<double> ToolPose = (cv::Mat_<double>(18, 6) <<
	-0.47262094,  0.15586345,  0.31719838, -1.92726779, -2.47331381, -0.02081063,
	-0.45740247,  0.12185793, 0.31721222, -2.1941084, -2.23784006, -0.01961464,
	-0.54939986,  0.12193493,  0.31726776, -2.19401449, -2.23764807, -0.01962044,
	-0.55660827,  0.26705787,  0.31725727, -2.19402042, -2.23774502, -0.01961607,
	-0.50059381,  0.27102397,  0.31718615, -2.19417578, -2.23791585, -0.01942449,
	-0.48737426,  0.16110843,  0.31724233, -2.60065113, -1.74452559, -0.01681714,
	-0.54617517,  0.0614836,  0.31725664, -2.95506626, -1.02949705, -0.01224048,
	-0.58419053,  0.10598133,  0.31725137, -2.95495237, -1.02958923, -0.01217588,
	-0.48648367,  0.19351801,  0.31725604, -2.95491889, -1.02963535, -0.01228079,
	-0.42126019,  0.18289687, 0.31720742, -2.95504234, -1.02952854, -0.01229755,
	-0.48160197,  0.14537841,  0.32101764,  2.87209407,  1.00174074, -0.22540814,
	-0.54278199,  0.14531701,  0.3210208,  2.87198785,  1.00184326, -0.22544345,
	-0.42396111,  0.31214188,  0.32098306,  2.87187933,  1.00164604, -0.22544516,
	-0.35770812,  0.23930883,  0.32103137,  2.87197266,  1.00181882, -0.22547622,
	-0.5162291,   0.15834344,  0.32940502, -2.87580658, -1.00716358, -0.23684506,
	-0.5754205,  0.01076326, 0.32937268, -2.87591133, -1.00712662, -0.23680695,
	-0.54433476,  0.10698335,  0.33395878, -2.22895581, -2.07114201, -0.38099654,
	-0.52836378,  0.18050263,  0.33397597, -2.22875369, -2.0712213, -0.38098654);  // 06标定数据*/
Mat_<double> ToolPose = (cv::Mat_<double>(20, 6) <<
	-0.49835378,  0.18026526,  0.23105621,  2.6145799,   1.68607664,  0.01326149,
-0.55674343,  0.11648082,  0.23246009,  2.12185278,  2.26064302,  0.30315816,
-0.55870445,  0.18453935,  0.21937721,  2.7942613,   0.89932147,  0.20515106,
-0.56050729,  0.24389209,  0.21875263,  2.1537516,   1.83133115,  0.07113669,
-0.62092332,  0.13707266,  0.21802162,  2.54880365,  1.30122249,  0.2392684,
-0.55576242,  0.04824733,  0.23905145,  3.03267909,  1.21842147, 0.15594604,
-0.46274972,  0.19906769,  0.27230332,  2.15843329, 2.63357234, 0.19156694,
-0.32924146,  0.22335662,  0.25758365,  2.89293113,  0.30448089, -0.45894559,
-0.51714978,  0.30072944,  0.31548851, 2.0632013, 2.40070119, -0.05039658,
-0.4903953,   0.28561995,  0.30183679,  2.65198206,  1.41809074,  0.1760023,
-0.52765498,  0.01196056,  0.31866186, 3.1273503,  1.0736474,  0.04632872,
-0.41793951,  0.24367578,  0.28887902, 1.9366007,  2.73618549, -0.19771895,
-0.48196411,  0.33405414,  0.28884921, 1.02886255, 3.22232721, -0.07911749,
-0.50765964,  0.20264815,  0.30860083,  2.96937673,  0.53856292,  0.20745904,
-0.56984617,  0.21288488,  0.26135811,  1.66714045, 2.35729663, -0.09245728,
-0.58277104,  0.13353368,  0.30545911,  2.97068042,  0.19301497,  0.11847784,
-0.49108837,  0.11874885,  0.37920522, 3.11685052, 0.92390712, -0.23202022,
-0.37913543,  0.07917403,  0.35593816, 2.17032489, 2.45895205, -0.21349414,
-0.46115461,  0.1961118,  0.33077234,  2.90664351,  0.68784452, -0.06574012,
-0.58927869,  0.24916414,  0.35974828,  1.84905267,  2.52647221,  0.25412332);


int main(int argc, char** argv)
{
	//数据声明
	vector<Mat> R_gripper2base;
	vector<Mat> T_gripper2base;
	vector<Mat> R_target2cam;
	vector<Mat> T_target2cam;
	Mat R_cam2gripper = Mat(3, 3, CV_64FC1);				//相机与机械臂末端坐标系的旋转矩阵与平移矩阵
	Mat T_cam2gripper = Mat(3, 1, CV_64FC1);
	Mat Homo_cam2gripper = Mat(4, 4, CV_64FC1);

	vector<Mat> Homo_target2cam;
	vector<Mat> Homo_gripper2base;
	Mat tempR, tempT, temp;

	/*首先求解相机的外参（目标在相机坐标系中的位姿）*/
	Mat chessImage;

	Size board_size = Size(9, 6);                         // 标定板上每行、列的角点数
	Size square_size = Size(25, 25);                       // 实际测量得到的标定板上每个棋盘格的物理尺寸，单位mm


	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));        // 摄像机内参数矩阵
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));          // 摄像机的5个畸变系数：k1,k2,p1,p2,k3
	vector<Mat> rvecsMat;                                          // 存放所有图像的旋转向量，每一副图像的旋转向量为一个mat
	vector<Mat> tvecsMat;                                          // 存放所有图像的平移向量，每一副图像的平移向量为一个mat
	vector<Mat> rvecsMat0;                                          // 存放所有图像的旋转向量，每一副图像的旋转向量为一个mat
	vector<Mat> tvecsMat0;                                          // 存放所有图像的平移向量，每一副图像的平移向量为一个mat
	Mat calpose;
	vector<Mat> CalPose;

	vector<String> imagesPath;//创建容器存放读取图像路径

	string image_path = "D:/graduate/handeyecali/Fri_Nov_25_17-19-40_2022/Fri_Nov_25_17-19-40_2022/*.bmp";//待处理图像路径	
	glob(image_path, imagesPath);//读取指定文件夹下图像

	m_calibration(imagesPath, board_size, square_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, rvecsMat0, tvecsMat0);  // 相机内外参标定

	for (int i = 0; i < tvecsMat.size(); i++)
	{
		vconcat(tvecsMat[i]*0.001, rvecsMat[i], calpose);
		cout << "相机的外参" << i << ": " << calpose.t() << endl;
		CalPose.push_back(calpose.t());
	}

	for (int i = 0; i < CalPose.size(); i++)				//计算标定板与相机间的齐次矩阵（旋转矩阵与平移向量）
	{
		//temp = attitudeVectorToMatrix(CalPose.row(i), false, "");	//注意seq为空，相机与标定板间的为旋转向量
		temp = attitudeVectorToMatrix(CalPose[i], false, "");
		Homo_target2cam.push_back(temp);
		HomogeneousMtr2RT(temp, tempR, tempT);
		/*cout << i << "::" << temp << endl;
		cout << i << "::" << tempR << endl;
		cout << i << "::" << tempT << endl;*/
		R_target2cam.push_back(tempR);
		T_target2cam.push_back(tempT);
	}
	for (int j = 0; j < ToolPose.rows; j++)				//计算机械臂末端坐标系与机器人基坐标系之间的齐次矩阵
	{
		temp = attitudeVectorToMatrix(ToolPose.row(j), false, "");  //注意seq不是空，机械臂末端坐标系与机器人基坐标系之间的为欧拉角
		Homo_gripper2base.push_back(temp);
		HomogeneousMtr2RT(temp, tempR, tempT);
		/*cout << j << "::" << temp << endl;
		cout << j << "::" << tempR << endl;
		cout << j << "::" << tempT << endl;*/
		R_gripper2base.push_back(tempR);
		T_gripper2base.push_back(tempT);
	}
	//TSAI计算速度最快
	calibrateHandEye(R_gripper2base, T_gripper2base, R_target2cam, T_target2cam, R_cam2gripper, T_cam2gripper, CALIB_HAND_EYE_TSAI);

	Homo_cam2gripper = R_T2HomogeneousMatrix(R_cam2gripper, T_cam2gripper);
	cout << Homo_cam2gripper << endl;
	cout << "Homo_cam2gripper 是否包含旋转矩阵：" << isRotatedMatrix(Homo_cam2gripper) << endl;

	///

		/**************************************************
		* @note   手眼系统精度测试，原理是标定板在机器人基坐标系中位姿固定不变，
		*		  可以根据这一点进行演算
		**************************************************/
		//使用1,2组数据验证  标定板在机器人基坐标系中位姿固定不变
	cout << "1 : " << Homo_gripper2base[0] * Homo_cam2gripper * Homo_target2cam[0] << endl;
	cout << "2 : " << Homo_gripper2base[1] * Homo_cam2gripper * Homo_target2cam[1] << endl;
	//标定板在相机中的位姿
	cout << "3 : " << Homo_target2cam[1] << endl;
	cout << "4 : " << Homo_cam2gripper.inv() * Homo_gripper2base[1].inv() * Homo_gripper2base[0] * Homo_cam2gripper * Homo_target2cam[0] << endl;

	cout << "----手眼系统测试-----" << endl;
	cout << "机械臂下标定板XYZ为：" << endl;
	for (int i = 0; i < Homo_target2cam.size(); i++)
	{
		Mat chessPos{ 0.0,0.0,0.0,1.0 };  //4*1矩阵，单独求机械臂坐标系下，标定板XYZ
		Mat worldPos = Homo_gripper2base[i] * Homo_cam2gripper * Homo_target2cam[i] * chessPos;
		cout << i << ": " << worldPos.t() << endl;
	}
	waitKey(0);

	return 0;
}

/**************************************************
* @brief   将旋转矩阵与平移向量合成为齐次矩阵
* @note
* @param   Mat& R   3*3旋转矩阵
* @param   Mat& T   3*1平移矩阵
* @return  Mat      4*4齐次矩阵
**************************************************/
Mat R_T2HomogeneousMatrix(const Mat& R, const Mat& T)
{
	Mat HomoMtr;
	Mat_<double> R1 = (Mat_<double>(4, 3) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
		0, 0, 0);
	Mat_<double> T1 = (Mat_<double>(4, 1) <<
		T.at<double>(0, 0),
		T.at<double>(1, 0),
		T.at<double>(2, 0),
		1);
	cv::hconcat(R1, T1, HomoMtr);		//矩阵拼接
	return HomoMtr;
}

/**************************************************
* @brief    齐次矩阵分解为旋转矩阵与平移矩阵
* @note
* @param	const Mat& HomoMtr  4*4齐次矩阵
* @param	Mat& R              输出旋转矩阵
* @param	Mat& T				输出平移矩阵
* @return
**************************************************/
void HomogeneousMtr2RT(Mat& HomoMtr, Mat& R, Mat& T)
{
	//Mat R_HomoMtr = HomoMtr(Rect(0, 0, 3, 3)); //注意Rect取值
	//Mat T_HomoMtr = HomoMtr(Rect(3, 0, 1, 3));
	//R_HomoMtr.copyTo(R);
	//T_HomoMtr.copyTo(T);
	/*HomoMtr(Rect(0, 0, 3, 3)).copyTo(R);
	HomoMtr(Rect(3, 0, 1, 3)).copyTo(T);*/
	Rect R_rect(0, 0, 3, 3);
	Rect T_rect(3, 0, 1, 3);
	R = HomoMtr(R_rect);
	T = HomoMtr(T_rect);

}

/**************************************************
* @brief	检查是否是旋转矩阵
* @note
* @param
* @param
* @param
* @return  true : 是旋转矩阵， false : 不是旋转矩阵
**************************************************/
bool isRotatedMatrix(Mat& R)		//旋转矩阵的转置矩阵是它的逆矩阵，逆矩阵 * 矩阵 = 单位矩阵
{
	Mat temp33 = R({ 0,0,3,3 });	//无论输入是几阶矩阵，均提取它的三阶矩阵
	Mat Rt;
	transpose(temp33, Rt);  //转置矩阵
	Mat shouldBeIdentity = Rt * temp33;//是旋转矩阵则乘积为单位矩阵
	Mat I = Mat::eye(3, 3, shouldBeIdentity.type());

	return cv::norm(I, shouldBeIdentity) < 1e-6;
}

/**************************************************
* @brief   欧拉角转换为旋转矩阵
* @note
* @param    const std::string& seq  指定欧拉角的排列顺序；（机械臂的位姿类型有xyz,zyx,zyz几种，需要区分）
* @param    const Mat& eulerAngle   欧拉角（1*3矩阵）, 角度值
* @param
* @return   返回3*3旋转矩阵
**************************************************/
Mat eulerAngleToRotateMatrix(const Mat& eulerAngle, const std::string& seq)
{
	CV_Assert(eulerAngle.rows == 1 && eulerAngle.cols == 3);//检查参数是否正确

	eulerAngle /= (180 / CV_PI);		//度转弧度

	Matx13d m(eulerAngle);				//<double, 1, 3>

	auto rx = m(0, 0), ry = m(0, 1), rz = m(0, 2);
	auto rxs = sin(rx), rxc = cos(rx);
	auto rys = sin(ry), ryc = cos(ry);
	auto rzs = sin(rz), rzc = cos(rz);

	//XYZ方向的旋转矩阵
	Mat RotX = (Mat_<double>(3, 3) << 1, 0, 0,
		0, rxc, -rxs,
		0, rxs, rxc);
	Mat RotY = (Mat_<double>(3, 3) << ryc, 0, rys,
		0, 1, 0,
		-rys, 0, ryc);
	Mat RotZ = (Mat_<double>(3, 3) << rzc, -rzs, 0,
		rzs, rzc, 0,
		0, 0, 1);
	//按顺序合成后的旋转矩阵
	cv::Mat rotMat;

	if (seq == "zyx") rotMat = RotX * RotY * RotZ;
	else if (seq == "yzx") rotMat = RotX * RotZ * RotY;
	else if (seq == "zxy") rotMat = RotY * RotX * RotZ;
	else if (seq == "yxz") rotMat = RotZ * RotX * RotY;
	else if (seq == "xyz") rotMat = RotZ * RotY * RotX;
	else if (seq == "xzy") rotMat = RotY * RotZ * RotX;
	else
	{
		cout << "Euler Angle Sequence string is wrong...";
	}
	if (!isRotatedMatrix(rotMat))		//欧拉角特殊情况下会出现死锁
	{
		cout << "Euler Angle convert to RotatedMatrix failed..." << endl;
		exit(-1);
	}
	return rotMat;
}

/**************************************************
* @brief   将四元数转换为旋转矩阵
* @note
* @param   const Vec4d& q   归一化的四元数: q = q0 + q1 * i + q2 * j + q3 * k;
* @return  返回3*3旋转矩阵R
**************************************************/
Mat quaternionToRotatedMatrix(const Vec4d& q)
{
	double q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];

	double q0q0 = q0 * q0, q1q1 = q1 * q1, q2q2 = q2 * q2, q3q3 = q3 * q3;
	double q0q1 = q0 * q1, q0q2 = q0 * q2, q0q3 = q0 * q3;
	double q1q2 = q1 * q2, q1q3 = q1 * q3;
	double q2q3 = q2 * q3;
	//根据公式得来
	Mat RotMtr = (Mat_<double>(3, 3) << (q0q0 + q1q1 - q2q2 - q3q3), 2 * (q1q2 + q0q3), 2 * (q1q3 - q0q2),
		2 * (q1q2 - q0q3), (q0q0 - q1q1 + q2q2 - q3q3), 2 * (q2q3 + q0q1),
		2 * (q1q3 + q0q2), 2 * (q2q3 - q0q1), (q0q0 - q1q1 - q2q2 + q3q3));
	//这种形式等价
	/*Mat RotMtr = (Mat_<double>(3, 3) << (1 - 2 * (q2q2 + q3q3)), 2 * (q1q2 - q0q3), 2 * (q1q3 + q0q2),
										 2 * (q1q2 + q0q3), 1 - 2 * (q1q1 + q3q3), 2 * (q2q3 - q0q1),
										 2 * (q1q3 - q0q2), 2 * (q2q3 + q0q1), (1 - 2 * (q1q1 + q2q2)));*/

	return RotMtr;
}

/**************************************************
* @brief      将采集的原始数据转换为齐次矩阵（从机器人控制器中获得的）
* @note
* @param	  Mat& m    1*6//1*10矩阵 ， 元素为： x,y,z,rx,ry,rz  or x,y,z, q0,q1,q2,q3,rx,ry,rz
* @param	  bool useQuaternion      原始数据是否使用四元数表示
* @param	  string& seq         原始数据使用欧拉角表示时，坐标系的旋转顺序
* @return	  返回转换完的齐次矩阵
**************************************************/
Mat attitudeVectorToMatrix(const Mat& m, bool useQuaternion, const string& seq)
{
	CV_Assert(m.total() == 6 || m.total() == 10);
	//if (m.cols == 1)	//转置矩阵为行矩阵
	//	m = m.t();	

	Mat temp = Mat::eye(4, 4, CV_64FC1);

	if (useQuaternion)
	{
		Vec4d quaternionVec = m({ 3,0,4,1 });   //读取存储的四元数
		quaternionToRotatedMatrix(quaternionVec).copyTo(temp({ 0,0,3,3 }));
	}
	else
	{
		Mat rotVec;
		if (m.total() == 6)
		{
			rotVec = m({ 3,0,3,1 });   //读取存储的欧拉角
		}
		if (m.total() == 10)
		{
			rotVec = m({ 7,0,3,1 });  //这边是不是有问题呢？
		}
		//如果seq为空，表示传入的是3*1旋转向量，否则，传入的是欧拉角
		if (0 == seq.compare(""))
		{
			Rodrigues(rotVec, temp({ 0,0,3,3 }));   //罗德利斯转换
		}
		else
		{
			eulerAngleToRotateMatrix(rotVec, seq).copyTo(temp({ 0,0,3,3 }));
		}
	}
	//存入平移矩阵
	temp({ 3,0,1,3 }) = m({ 0,0,3,1 }).t();
	return temp;   //返回转换结束的齐次矩阵
}


void m_calibration(vector<string>& FilesName, Size board_size, Size square_size, Mat& cameraMatrix, Mat& distCoeffs, vector<Mat>& rvecsMat, vector<Mat>& tvecsMat, vector<Mat>& rvecsMat0, vector<Mat>& tvecsMat0)
{
	ofstream fout_0("calibration_result(solvepnp).txt");
	ofstream fout("calibration_result.txt");                       // 保存标定结果的文件

	cout << "开始提取角点………………" << endl;
	int image_count = 0;                                            // 图像数量
	Size image_size;                                                // 图像的尺寸

	vector<Point2f> image_points;                                   // 缓存每幅图像上检测到的角点
	vector<vector<Point2f>> image_points_seq;                       // 保存检测到的所有角点
	
	for (int i = 0; i < FilesName.size(); i++)
	{
		image_count++;

		// 用于观察检验输出
		cout << "image_count = " << image_count << endl;
		Mat imageInput = imread(FilesName[i]);
		if (image_count == 1)  //读入第一张图片时获取图像宽高信息
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			cout << "image_size.width = " << image_size.width << endl;
			cout << "image_size.height = " << image_size.height << endl;
		}

		/* 提取角点 */
		bool bRes = findChessboardCorners(imageInput, board_size, image_points, 0);
		if (bRes)
		{
			Mat view_gray;
			cout << "imageInput.channels()=" << imageInput.channels() << endl;
			cvtColor(imageInput, view_gray, cv::COLOR_RGB2GRAY);

			/* 亚像素精确化 */
			cv::cornerSubPix(view_gray, image_points, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.01));

			image_points_seq.push_back(image_points);  //保存亚像素角点

			/* 在图像上显示角点位置 */
			drawChessboardCorners(view_gray, board_size, image_points, true);

			imshow("Camera Calibration", view_gray);//显示图片
			waitKey(0);//暂停
		}
		else
		{
			cout << "第" << image_count << "张照片提取角点失败，请删除后，重新标定！" << endl; //找不到角点
			imshow("失败照片", imageInput);
			waitKey(0);
		}
	}
	cout << "角点提取完成！！！" << endl;


	/*棋盘三维信息*/
	vector<vector<Point3f>> object_points_seq;                     // 保存标定板上角点的三维坐标

	for (int t = 0; t < image_count; t++)
	{
		vector<Point3f> object_points;
		for (int i = 0; i < board_size.height; i++)
		{
			for (int j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;
				/* 假设标定板放在世界坐标系中z=0的平面上 */
				realPoint.x = i * square_size.width;
				realPoint.y = j * square_size.height;
				realPoint.z = 0;
				object_points.push_back(realPoint);
			}
		}
		object_points_seq.push_back(object_points);
	}

	double distCoeffD[5] = {0,0,0,0,0};

	Mat cameraMatrix_0(Matx33d(607.2985229492188, 0, 315.4208984375,
		0, 607.184326171875, 233.615478515625,
		0, 0, 1));
	Mat distCoeffs_0 = Mat(1, 5, CV_64FC1, distCoeffD);          // 摄像机的5个畸变系数：k1,k2,p1,p2,k3
	Mat rvecs;                                          // 存放一张图像的旋转向量，每一副图像的旋转向量为一个mat
	Mat tvecs;                                          // 存放一张图像的平移向量，每一副图像的平移向量为一个mat
	//vector<Mat> rvecsMat0;
	//vector<Mat> tvecsMat0;

	cout << "************solvePnP准备运行*************" << endl;
	fout_0 << "相机内参数矩阵：" << endl;
	fout_0 << cameraMatrix_0 << endl << endl;
	fout_0 << "畸变系数：\n";
	fout_0 << distCoeffs_0 << endl << endl << endl;
	for (int i = 0; i < image_count; i++)
	{
		bool ret = cv::solvePnP(object_points_seq[i], image_points_seq[i], cameraMatrix_0, distCoeffs_0, rvecs, tvecs);
		fout_0 << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout_0 << rvecs << endl;
		rvecsMat0.push_back(rvecs);

		fout_0 << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout_0 << tvecs << endl << endl;
		tvecsMat0.push_back(tvecs);

	}
	/* 运行标定函数 */
	double rms = calibrateCamera(object_points_seq, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST);
	cout << "RMS：" << rms << "像素" << endl << endl;
	cout << "标定完成！！！" << endl;

	cout << "开始评价标定结果………………";
	double total_err = 0.0;            // 所有图像的平均误差的总和
	double err = 0.0;                  // 每幅图像的平均误差
	double totalErr = 0.0;
	double totalPoints = 0.0;
	double total_err_0 = 0.0;            // solvepnp所有图像的平均误差的总和
	double err_0 = 0.0;                  // solvepnp每幅图像的平均误差
	double totalErr_0 = 0.0;
	vector<Point2f> image_points_pro;     // 保存重新计算得到的投影点
	vector<Point2f> image_points_pro_0;     // 保存solvepnp重新计算得到的投影点


	for (int i = 0; i < image_count; i++)
	{
		projectPoints(object_points_seq[i], rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points_pro);   //通过得到的摄像机内外参数，对角点的空间三维坐标进行重新投影计算
		projectPoints(object_points_seq[i], rvecsMat0[i], tvecsMat0[i], cameraMatrix_0, distCoeffs_0, image_points_pro_0);
		
		err = norm(Mat(image_points_seq[i]), Mat(image_points_pro), NORM_L2);
		err_0 = norm(Mat(image_points_seq[i]), Mat(image_points_pro_0), NORM_L2);

		totalErr += err * err;
		totalErr_0 += err_0 * err_0;
		totalPoints += object_points_seq[i].size();

		err /= object_points_seq[i].size();
		err_0 /= object_points_seq[i].size();
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout_0 << "第" << i + 1 << "幅图像的平均误差(solvepnp)：" << err_0 << "像素" << endl;
		total_err += err;
		total_err_0 += err_0;
	}
	fout << "重投影误差2：" << sqrt(totalErr / totalPoints) << "像素" << endl << endl;
	fout << "重投影误差3：" << total_err / image_count << "像素" << endl << endl;
	fout_0 << "重投影误差2：" << sqrt(totalErr_0 / totalPoints) << "像素" << endl << endl;
	fout_0 << "重投影误差3：" << total_err_0 / image_count << "像素" << endl << endl;
	cout << "x = " << cameraMatrix.at<double>(0, 2) << endl;
	cout << "y = " << cameraMatrix.at<double>(1, 2) << endl;
	cout << "x_0 = " << cameraMatrix_0.at<double>(0, 2) << endl;
	cout << "y_0 = " << cameraMatrix_0.at<double>(1, 2) << endl;
	
	//保存定标结果
	cout << "开始保存定标结果………………" << endl;
	//Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
	fout << "相机内参数矩阵：" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "畸变系数：\n";
	fout << distCoeffs << endl << endl << endl;

	for (int i = 0; i < image_count; i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
	    fout << rvecsMat[i] << endl;

	//     /* 将旋转向量转换为相对应的旋转矩阵 */
	//	   Rodrigues(rvecsMat[i], rotation_matrix);
	//     fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
	//     fout << rotation_matrix << endl;
	   fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
	   fout << tvecsMat[i] << endl << endl;

	}
	cout << "标定标结果完成保存！！！" << endl;
	fout << endl;
}



