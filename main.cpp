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



//����ʹ�õ����е�ֵ
//�����13��궨���λ�ˣ�x,y,z��rx,ry,rz,
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
-0.49721658,  0.18193714,  0.26989715,  2.87110418,  1.00354908, -0.1663514); //04�궨����
*/
//��е��ĩ��13��λ��,x,y,z,rx,ry,rz
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
	-0.52836378,  0.18050263,  0.33397597, -2.22875369, -2.0712213, -0.38098654);  // 06�궨����*/
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
	//��������
	vector<Mat> R_gripper2base;
	vector<Mat> T_gripper2base;
	vector<Mat> R_target2cam;
	vector<Mat> T_target2cam;
	Mat R_cam2gripper = Mat(3, 3, CV_64FC1);				//������е��ĩ������ϵ����ת������ƽ�ƾ���
	Mat T_cam2gripper = Mat(3, 1, CV_64FC1);
	Mat Homo_cam2gripper = Mat(4, 4, CV_64FC1);

	vector<Mat> Homo_target2cam;
	vector<Mat> Homo_gripper2base;
	Mat tempR, tempT, temp;

	/*��������������Σ�Ŀ�����������ϵ�е�λ�ˣ�*/
	Mat chessImage;

	Size board_size = Size(9, 6);                         // �궨����ÿ�С��еĽǵ���
	Size square_size = Size(25, 25);                       // ʵ�ʲ����õ��ı궨����ÿ�����̸������ߴ磬��λmm


	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));        // ������ڲ�������
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));          // �������5������ϵ����k1,k2,p1,p2,k3
	vector<Mat> rvecsMat;                                          // �������ͼ�����ת������ÿһ��ͼ�����ת����Ϊһ��mat
	vector<Mat> tvecsMat;                                          // �������ͼ���ƽ��������ÿһ��ͼ���ƽ������Ϊһ��mat
	vector<Mat> rvecsMat0;                                          // �������ͼ�����ת������ÿһ��ͼ�����ת����Ϊһ��mat
	vector<Mat> tvecsMat0;                                          // �������ͼ���ƽ��������ÿһ��ͼ���ƽ������Ϊһ��mat
	Mat calpose;
	vector<Mat> CalPose;

	vector<String> imagesPath;//����������Ŷ�ȡͼ��·��

	string image_path = "D:/graduate/handeyecali/Fri_Nov_25_17-19-40_2022/Fri_Nov_25_17-19-40_2022/*.bmp";//������ͼ��·��	
	glob(image_path, imagesPath);//��ȡָ���ļ�����ͼ��

	m_calibration(imagesPath, board_size, square_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, rvecsMat0, tvecsMat0);  // �������α궨

	for (int i = 0; i < tvecsMat.size(); i++)
	{
		vconcat(tvecsMat[i]*0.001, rvecsMat[i], calpose);
		cout << "��������" << i << ": " << calpose.t() << endl;
		CalPose.push_back(calpose.t());
	}

	for (int i = 0; i < CalPose.size(); i++)				//����궨������������ξ�����ת������ƽ��������
	{
		//temp = attitudeVectorToMatrix(CalPose.row(i), false, "");	//ע��seqΪ�գ������궨����Ϊ��ת����
		temp = attitudeVectorToMatrix(CalPose[i], false, "");
		Homo_target2cam.push_back(temp);
		HomogeneousMtr2RT(temp, tempR, tempT);
		/*cout << i << "::" << temp << endl;
		cout << i << "::" << tempR << endl;
		cout << i << "::" << tempT << endl;*/
		R_target2cam.push_back(tempR);
		T_target2cam.push_back(tempT);
	}
	for (int j = 0; j < ToolPose.rows; j++)				//�����е��ĩ������ϵ������˻�����ϵ֮�����ξ���
	{
		temp = attitudeVectorToMatrix(ToolPose.row(j), false, "");  //ע��seq���ǿգ���е��ĩ������ϵ������˻�����ϵ֮���Ϊŷ����
		Homo_gripper2base.push_back(temp);
		HomogeneousMtr2RT(temp, tempR, tempT);
		/*cout << j << "::" << temp << endl;
		cout << j << "::" << tempR << endl;
		cout << j << "::" << tempT << endl;*/
		R_gripper2base.push_back(tempR);
		T_gripper2base.push_back(tempT);
	}
	//TSAI�����ٶ����
	calibrateHandEye(R_gripper2base, T_gripper2base, R_target2cam, T_target2cam, R_cam2gripper, T_cam2gripper, CALIB_HAND_EYE_TSAI);

	Homo_cam2gripper = R_T2HomogeneousMatrix(R_cam2gripper, T_cam2gripper);
	cout << Homo_cam2gripper << endl;
	cout << "Homo_cam2gripper �Ƿ������ת����" << isRotatedMatrix(Homo_cam2gripper) << endl;

	///

		/**************************************************
		* @note   ����ϵͳ���Ȳ��ԣ�ԭ���Ǳ궨���ڻ����˻�����ϵ��λ�˹̶����䣬
		*		  ���Ը�����һ���������
		**************************************************/
		//ʹ��1,2��������֤  �궨���ڻ����˻�����ϵ��λ�˹̶�����
	cout << "1 : " << Homo_gripper2base[0] * Homo_cam2gripper * Homo_target2cam[0] << endl;
	cout << "2 : " << Homo_gripper2base[1] * Homo_cam2gripper * Homo_target2cam[1] << endl;
	//�궨��������е�λ��
	cout << "3 : " << Homo_target2cam[1] << endl;
	cout << "4 : " << Homo_cam2gripper.inv() * Homo_gripper2base[1].inv() * Homo_gripper2base[0] * Homo_cam2gripper * Homo_target2cam[0] << endl;

	cout << "----����ϵͳ����-----" << endl;
	cout << "��е���±궨��XYZΪ��" << endl;
	for (int i = 0; i < Homo_target2cam.size(); i++)
	{
		Mat chessPos{ 0.0,0.0,0.0,1.0 };  //4*1���󣬵������е������ϵ�£��궨��XYZ
		Mat worldPos = Homo_gripper2base[i] * Homo_cam2gripper * Homo_target2cam[i] * chessPos;
		cout << i << ": " << worldPos.t() << endl;
	}
	waitKey(0);

	return 0;
}

/**************************************************
* @brief   ����ת������ƽ�������ϳ�Ϊ��ξ���
* @note
* @param   Mat& R   3*3��ת����
* @param   Mat& T   3*1ƽ�ƾ���
* @return  Mat      4*4��ξ���
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
	cv::hconcat(R1, T1, HomoMtr);		//����ƴ��
	return HomoMtr;
}

/**************************************************
* @brief    ��ξ���ֽ�Ϊ��ת������ƽ�ƾ���
* @note
* @param	const Mat& HomoMtr  4*4��ξ���
* @param	Mat& R              �����ת����
* @param	Mat& T				���ƽ�ƾ���
* @return
**************************************************/
void HomogeneousMtr2RT(Mat& HomoMtr, Mat& R, Mat& T)
{
	//Mat R_HomoMtr = HomoMtr(Rect(0, 0, 3, 3)); //ע��Rectȡֵ
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
* @brief	����Ƿ�����ת����
* @note
* @param
* @param
* @param
* @return  true : ����ת���� false : ������ת����
**************************************************/
bool isRotatedMatrix(Mat& R)		//��ת�����ת�þ������������������� * ���� = ��λ����
{
	Mat temp33 = R({ 0,0,3,3 });	//���������Ǽ��׾��󣬾���ȡ�������׾���
	Mat Rt;
	transpose(temp33, Rt);  //ת�þ���
	Mat shouldBeIdentity = Rt * temp33;//����ת������˻�Ϊ��λ����
	Mat I = Mat::eye(3, 3, shouldBeIdentity.type());

	return cv::norm(I, shouldBeIdentity) < 1e-6;
}

/**************************************************
* @brief   ŷ����ת��Ϊ��ת����
* @note
* @param    const std::string& seq  ָ��ŷ���ǵ�����˳�򣻣���е�۵�λ��������xyz,zyx,zyz���֣���Ҫ���֣�
* @param    const Mat& eulerAngle   ŷ���ǣ�1*3����, �Ƕ�ֵ
* @param
* @return   ����3*3��ת����
**************************************************/
Mat eulerAngleToRotateMatrix(const Mat& eulerAngle, const std::string& seq)
{
	CV_Assert(eulerAngle.rows == 1 && eulerAngle.cols == 3);//�������Ƿ���ȷ

	eulerAngle /= (180 / CV_PI);		//��ת����

	Matx13d m(eulerAngle);				//<double, 1, 3>

	auto rx = m(0, 0), ry = m(0, 1), rz = m(0, 2);
	auto rxs = sin(rx), rxc = cos(rx);
	auto rys = sin(ry), ryc = cos(ry);
	auto rzs = sin(rz), rzc = cos(rz);

	//XYZ�������ת����
	Mat RotX = (Mat_<double>(3, 3) << 1, 0, 0,
		0, rxc, -rxs,
		0, rxs, rxc);
	Mat RotY = (Mat_<double>(3, 3) << ryc, 0, rys,
		0, 1, 0,
		-rys, 0, ryc);
	Mat RotZ = (Mat_<double>(3, 3) << rzc, -rzs, 0,
		rzs, rzc, 0,
		0, 0, 1);
	//��˳��ϳɺ����ת����
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
	if (!isRotatedMatrix(rotMat))		//ŷ������������»��������
	{
		cout << "Euler Angle convert to RotatedMatrix failed..." << endl;
		exit(-1);
	}
	return rotMat;
}

/**************************************************
* @brief   ����Ԫ��ת��Ϊ��ת����
* @note
* @param   const Vec4d& q   ��һ������Ԫ��: q = q0 + q1 * i + q2 * j + q3 * k;
* @return  ����3*3��ת����R
**************************************************/
Mat quaternionToRotatedMatrix(const Vec4d& q)
{
	double q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];

	double q0q0 = q0 * q0, q1q1 = q1 * q1, q2q2 = q2 * q2, q3q3 = q3 * q3;
	double q0q1 = q0 * q1, q0q2 = q0 * q2, q0q3 = q0 * q3;
	double q1q2 = q1 * q2, q1q3 = q1 * q3;
	double q2q3 = q2 * q3;
	//���ݹ�ʽ����
	Mat RotMtr = (Mat_<double>(3, 3) << (q0q0 + q1q1 - q2q2 - q3q3), 2 * (q1q2 + q0q3), 2 * (q1q3 - q0q2),
		2 * (q1q2 - q0q3), (q0q0 - q1q1 + q2q2 - q3q3), 2 * (q2q3 + q0q1),
		2 * (q1q3 + q0q2), 2 * (q2q3 - q0q1), (q0q0 - q1q1 - q2q2 + q3q3));
	//������ʽ�ȼ�
	/*Mat RotMtr = (Mat_<double>(3, 3) << (1 - 2 * (q2q2 + q3q3)), 2 * (q1q2 - q0q3), 2 * (q1q3 + q0q2),
										 2 * (q1q2 + q0q3), 1 - 2 * (q1q1 + q3q3), 2 * (q2q3 - q0q1),
										 2 * (q1q3 - q0q2), 2 * (q2q3 + q0q1), (1 - 2 * (q1q1 + q2q2)));*/

	return RotMtr;
}

/**************************************************
* @brief      ���ɼ���ԭʼ����ת��Ϊ��ξ��󣨴ӻ����˿������л�õģ�
* @note
* @param	  Mat& m    1*6//1*10���� �� Ԫ��Ϊ�� x,y,z,rx,ry,rz  or x,y,z, q0,q1,q2,q3,rx,ry,rz
* @param	  bool useQuaternion      ԭʼ�����Ƿ�ʹ����Ԫ����ʾ
* @param	  string& seq         ԭʼ����ʹ��ŷ���Ǳ�ʾʱ������ϵ����ת˳��
* @return	  ����ת�������ξ���
**************************************************/
Mat attitudeVectorToMatrix(const Mat& m, bool useQuaternion, const string& seq)
{
	CV_Assert(m.total() == 6 || m.total() == 10);
	//if (m.cols == 1)	//ת�þ���Ϊ�о���
	//	m = m.t();	

	Mat temp = Mat::eye(4, 4, CV_64FC1);

	if (useQuaternion)
	{
		Vec4d quaternionVec = m({ 3,0,4,1 });   //��ȡ�洢����Ԫ��
		quaternionToRotatedMatrix(quaternionVec).copyTo(temp({ 0,0,3,3 }));
	}
	else
	{
		Mat rotVec;
		if (m.total() == 6)
		{
			rotVec = m({ 3,0,3,1 });   //��ȡ�洢��ŷ����
		}
		if (m.total() == 10)
		{
			rotVec = m({ 7,0,3,1 });  //����ǲ����������أ�
		}
		//���seqΪ�գ���ʾ�������3*1��ת���������򣬴������ŷ����
		if (0 == seq.compare(""))
		{
			Rodrigues(rotVec, temp({ 0,0,3,3 }));   //�޵���˹ת��
		}
		else
		{
			eulerAngleToRotateMatrix(rotVec, seq).copyTo(temp({ 0,0,3,3 }));
		}
	}
	//����ƽ�ƾ���
	temp({ 3,0,1,3 }) = m({ 0,0,3,1 }).t();
	return temp;   //����ת����������ξ���
}


void m_calibration(vector<string>& FilesName, Size board_size, Size square_size, Mat& cameraMatrix, Mat& distCoeffs, vector<Mat>& rvecsMat, vector<Mat>& tvecsMat, vector<Mat>& rvecsMat0, vector<Mat>& tvecsMat0)
{
	ofstream fout_0("calibration_result(solvepnp).txt");
	ofstream fout("calibration_result.txt");                       // ����궨������ļ�

	cout << "��ʼ��ȡ�ǵ㡭����������" << endl;
	int image_count = 0;                                            // ͼ������
	Size image_size;                                                // ͼ��ĳߴ�

	vector<Point2f> image_points;                                   // ����ÿ��ͼ���ϼ�⵽�Ľǵ�
	vector<vector<Point2f>> image_points_seq;                       // �����⵽�����нǵ�
	
	for (int i = 0; i < FilesName.size(); i++)
	{
		image_count++;

		// ���ڹ۲�������
		cout << "image_count = " << image_count << endl;
		Mat imageInput = imread(FilesName[i]);
		if (image_count == 1)  //�����һ��ͼƬʱ��ȡͼ������Ϣ
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			cout << "image_size.width = " << image_size.width << endl;
			cout << "image_size.height = " << image_size.height << endl;
		}

		/* ��ȡ�ǵ� */
		bool bRes = findChessboardCorners(imageInput, board_size, image_points, 0);
		if (bRes)
		{
			Mat view_gray;
			cout << "imageInput.channels()=" << imageInput.channels() << endl;
			cvtColor(imageInput, view_gray, cv::COLOR_RGB2GRAY);

			/* �����ؾ�ȷ�� */
			cv::cornerSubPix(view_gray, image_points, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.01));

			image_points_seq.push_back(image_points);  //���������ؽǵ�

			/* ��ͼ������ʾ�ǵ�λ�� */
			drawChessboardCorners(view_gray, board_size, image_points, true);

			imshow("Camera Calibration", view_gray);//��ʾͼƬ
			waitKey(0);//��ͣ
		}
		else
		{
			cout << "��" << image_count << "����Ƭ��ȡ�ǵ�ʧ�ܣ���ɾ�������±궨��" << endl; //�Ҳ����ǵ�
			imshow("ʧ����Ƭ", imageInput);
			waitKey(0);
		}
	}
	cout << "�ǵ���ȡ��ɣ�����" << endl;


	/*������ά��Ϣ*/
	vector<vector<Point3f>> object_points_seq;                     // ����궨���Ͻǵ����ά����

	for (int t = 0; t < image_count; t++)
	{
		vector<Point3f> object_points;
		for (int i = 0; i < board_size.height; i++)
		{
			for (int j = 0; j < board_size.width; j++)
			{
				Point3f realPoint;
				/* ����궨�������������ϵ��z=0��ƽ���� */
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
	Mat distCoeffs_0 = Mat(1, 5, CV_64FC1, distCoeffD);          // �������5������ϵ����k1,k2,p1,p2,k3
	Mat rvecs;                                          // ���һ��ͼ�����ת������ÿһ��ͼ�����ת����Ϊһ��mat
	Mat tvecs;                                          // ���һ��ͼ���ƽ��������ÿһ��ͼ���ƽ������Ϊһ��mat
	//vector<Mat> rvecsMat0;
	//vector<Mat> tvecsMat0;

	cout << "************solvePnP׼������*************" << endl;
	fout_0 << "����ڲ�������" << endl;
	fout_0 << cameraMatrix_0 << endl << endl;
	fout_0 << "����ϵ����\n";
	fout_0 << distCoeffs_0 << endl << endl << endl;
	for (int i = 0; i < image_count; i++)
	{
		bool ret = cv::solvePnP(object_points_seq[i], image_points_seq[i], cameraMatrix_0, distCoeffs_0, rvecs, tvecs);
		fout_0 << "��" << i + 1 << "��ͼ�����ת������" << endl;
		fout_0 << rvecs << endl;
		rvecsMat0.push_back(rvecs);

		fout_0 << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
		fout_0 << tvecs << endl << endl;
		tvecsMat0.push_back(tvecs);

	}
	/* ���б궨���� */
	double rms = calibrateCamera(object_points_seq, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST);
	cout << "RMS��" << rms << "����" << endl << endl;
	cout << "�궨��ɣ�����" << endl;

	cout << "��ʼ���۱궨���������������";
	double total_err = 0.0;            // ����ͼ���ƽ�������ܺ�
	double err = 0.0;                  // ÿ��ͼ���ƽ�����
	double totalErr = 0.0;
	double totalPoints = 0.0;
	double total_err_0 = 0.0;            // solvepnp����ͼ���ƽ�������ܺ�
	double err_0 = 0.0;                  // solvepnpÿ��ͼ���ƽ�����
	double totalErr_0 = 0.0;
	vector<Point2f> image_points_pro;     // �������¼���õ���ͶӰ��
	vector<Point2f> image_points_pro_0;     // ����solvepnp���¼���õ���ͶӰ��


	for (int i = 0; i < image_count; i++)
	{
		projectPoints(object_points_seq[i], rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points_pro);   //ͨ���õ������������������Խǵ�Ŀռ���ά�����������ͶӰ����
		projectPoints(object_points_seq[i], rvecsMat0[i], tvecsMat0[i], cameraMatrix_0, distCoeffs_0, image_points_pro_0);
		
		err = norm(Mat(image_points_seq[i]), Mat(image_points_pro), NORM_L2);
		err_0 = norm(Mat(image_points_seq[i]), Mat(image_points_pro_0), NORM_L2);

		totalErr += err * err;
		totalErr_0 += err_0 * err_0;
		totalPoints += object_points_seq[i].size();

		err /= object_points_seq[i].size();
		err_0 /= object_points_seq[i].size();
		fout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
		fout_0 << "��" << i + 1 << "��ͼ���ƽ�����(solvepnp)��" << err_0 << "����" << endl;
		total_err += err;
		total_err_0 += err_0;
	}
	fout << "��ͶӰ���2��" << sqrt(totalErr / totalPoints) << "����" << endl << endl;
	fout << "��ͶӰ���3��" << total_err / image_count << "����" << endl << endl;
	fout_0 << "��ͶӰ���2��" << sqrt(totalErr_0 / totalPoints) << "����" << endl << endl;
	fout_0 << "��ͶӰ���3��" << total_err_0 / image_count << "����" << endl << endl;
	cout << "x = " << cameraMatrix.at<double>(0, 2) << endl;
	cout << "y = " << cameraMatrix.at<double>(1, 2) << endl;
	cout << "x_0 = " << cameraMatrix_0.at<double>(0, 2) << endl;
	cout << "y_0 = " << cameraMatrix_0.at<double>(1, 2) << endl;
	
	//���涨����
	cout << "��ʼ���涨����������������" << endl;
	//Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* ����ÿ��ͼ�����ת���� */
	fout << "����ڲ�������" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "����ϵ����\n";
	fout << distCoeffs << endl << endl << endl;

	for (int i = 0; i < image_count; i++)
	{
		fout << "��" << i + 1 << "��ͼ�����ת������" << endl;
	    fout << rvecsMat[i] << endl;

	//     /* ����ת����ת��Ϊ���Ӧ����ת���� */
	//	   Rodrigues(rvecsMat[i], rotation_matrix);
	//     fout << "��" << i + 1 << "��ͼ�����ת����" << endl;
	//     fout << rotation_matrix << endl;
	   fout << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
	   fout << tvecsMat[i] << endl << endl;

	}
	cout << "�궨������ɱ��棡����" << endl;
	fout << endl;
}



