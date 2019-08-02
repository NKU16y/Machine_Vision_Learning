#include<highgui.hpp>
#include<core.hpp>
#include<opencv.hpp>
#include<cmath>
#include"cv.h"
using namespace std;
using namespace cv;
int main()
{
	CvCapture*capture = cvCreateFileCapture("F:\\课件\\机器视觉\\input.avi");//读取视频
	IplImage* mframe = cvQueryFrame(capture);//读取视频中的第一帧
	IplImage* current;//保存截取的图片灰度化结果
	IplImage*fore;//前景
	current = cvCreateImage(cvSize(mframe->width, mframe->height), IPL_DEPTH_8U, 1);
	fore = cvCreateImage(cvSize(mframe->width, mframe->height), IPL_DEPTH_8U, 1);
	cvCvtColor(mframe, current, CV_BGR2GRAY);//灰度化
	uchar* data = (uchar *)current->imageData;
	uchar* data2 = (uchar *)fore->imageData;

	int height = mframe->height;
	int width = mframe->width;

	VideoWriter writer("F:\\课件\\机器视觉\\output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
		Size(width, height), 1);

	int K, B;//B为检测背景所用数,K为高斯模型个数
	B = 0;
	K = 4;
	double alpha, Sigma0, Wint;//学习率α，初始方差sigma0,第一个高斯模型的初始权重Wint
	alpha=0.01;
	Sigma0=10;
	Wint=0.6;
	double lamba = 2.5;
	double T = 0.7;//检测背景所用阈值
	double *miu = new double[width*height*K];//均值
	double *sigma = new double[width*height*K];//标准差
	double *weight = new double[width*height*K];//权值  
	double *diff = new double[width*height*K];//像素与均值差值  
	double *rank = new double[K];//高斯函数排序
	//初始化
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)//对第一帧每个像素 
		{
			for (int k = 0; k < K; ++k) //对每一个高斯模型
			{
				if (k == 0)//对第一个高斯模型,初始化均值为第一帧图像像素值，权值为Wint
				{
					miu[i*width*K + j*K + k] = data[i*width + j];
					weight[i*width*K + j*K + k] = Wint;
				}
				else
				{//对其他高斯模型,均值为0，权重为(1-Wint)/(K-1)
					miu[i*width*K + j*K + k] = 0;
					weight[i*width*K + j*K + k] = double((1 - Wint) / (K - 1));
				}
				//所有高斯模型，标准差均为Sigma0
				sigma[i*width*K + j*K + k] = Sigma0;
			}
		}
	}
	//初始化完成
	int i = 0;
	int j = 0;
	int k = 0;
	while (1)//对每一帧图像
	{
		int *rank_ind = new int[K];
		cvCvtColor(mframe, current, CV_BGR2GRAY);
		//对于每一个像素，分别计算它和每一个单高斯模型的均值的差值
		for (i = 0; i < height; i++)//对于每一个像素
		{
			for (j = 0; j < width; j++)
			{
				for (k = 0; k < K; k++)
				{
					diff[i*width*K + j*K + k] = abs(data[i*width + j] - miu[i*width*K + j*K + k]);
				}//end  k
			}//end  j
		}//end  i

		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				int l = 100;
				bool match = 0;

				//遍历所有的单高斯模式，如果此像素满足任一单高斯模式，则匹配；如果此像素不满足任何的单高斯模式，则不匹配
				for (int k = 0; k < K; k++)
				{
					if (diff[i*width*K + j*K + k] <= lamba*sigma[i*width*K + j*K + k])//找到满足匹配的最小的高斯模型的编号
					{
						match = 1;
						l = k;
						break;
					}
				}
				//如果匹配
				if (match) {
					for (int k = 0; k < K; k++)//遍历所有的单高斯模式，更新权重和归一化
					{
						if (k == l)//对于满足匹配的最小的k即匹配的高斯模型，权重不变
						{
							weight[i*width*K + j*K + k] = weight[i*width*K + j*K + k];
						}
						else//对其他高斯模型，权重变为原来的(1-α)倍
						{
							weight[i*width*K + j*K + k] = (1 - alpha)*weight[i*width*K + j*K + k];
						}
					}
					//权值归一化
					double Mtemp = 0.0;
					for (k = 0; k < K; k++)
					{
						Mtemp += weight[i*width*K + j*K + k];//计算K个单高斯模式权值的和
					}
					for (k = 0; k < K; k++)
					{
						weight[i*width*K + j*K + k] = weight[i*width*K + j*K + k] / Mtemp;//权值归一化，使得所有权值和为1
					}
					//对匹配的高斯模型更新参数
					double Sum = 0.0;
					double p = 0.0;
					for (int k = 0; k < K; k++)
					{
						Sum += weight[i*width*K + j*K + k] * double(1 / sqrt(2 * 3.14))*pow(2.7183, -pow((data[i*width + j] - miu[i*width*K + j*K + k]) / sigma[i*width*K + j*K + k], 2));
					}
					//Sum计算概率值gt;
					p = alpha*Sum;
					miu[i*width*K + j*K + l] = (1 - p)*miu[i*width*K + j*K + l] + p*data[i*width + j];
					sigma[i*width*K + j*K + l] = sqrt(1 - p)*sigma[i*width*K + j*K + l] + sqrt(p)*abs(data[i*width + j] - miu[i*width*K + j*K + l]);

				}
				else//如果和所有单高斯模式都不匹配，则寻找最小权重和最大标准差.然后将最小权重替换为一个新的高斯模式
				{
					double one_temp = 0.0;
					double two_temp = 0.0;
					int tempnum = 0;
					one_temp = weight[i*width*K + j*K];
					two_temp = sigma[i*width*K + j*K];
					for (k = 0; k < K; k++)
					{
						if (weight[i*width*K + j*K + k] <= one_temp)
						{
							one_temp = weight[i*width*K + j*K + k];
							tempnum = k;
						}
						if (sigma[i*width*K + j*K + k] > two_temp)
						{
							two_temp = sigma[i*width*K + j*K + k];
						}
					}
					//找到最小权重和最大标准差
					//将权值最小的一个替换为新的
					miu[i*width*K + j*K + tempnum] = data[i*width + j];
					sigma[i*width*K + j*K + tempnum] = sqrt(2)*two_temp;
					weight[i*width*K + j*K + tempnum] = 0.5*one_temp;
					double NMtemp = 0.0;
					//权值归一化
					for (k = 0; k < K; k++)
					{
						NMtemp += weight[i*width*K + j*K + k];//计算四个单高斯模式权值的和
					}
					for (k = 0; k < K; k++)
					{
						weight[i*width*K + j*K + k] = weight[i*width*K + j*K + k] / NMtemp;//权值归一化，使得所有权值和为1
					}
				}

				for (k = 0; k < K; k++)//计算每个单高斯模式的重要性
				{
					rank[k] = weight[i*width*K + j*K + k] / sigma[i*width*K + j*K + k];
					rank_ind[k] = k;
				}
				for (k = 0; k < K; k++)//对重要性排序
				{
					for (int m = K - 1; m > k; m--)
					{
						double rank_temp = 0;
						int rank_ind_temp = 0;
						if (rank[m] > rank[m - 1])
						{
							//swap max values  
							rank_temp = rank[m];
							rank[m] = rank[m - 1];
							rank[m - 1] = rank_temp;
							//swap max index values  
							rank_ind_temp = rank_ind[m];
							rank_ind[m] = rank_ind[m - 1];
							rank_ind[m - 1] = rank_ind_temp;
						}
					}
				}
				double wtemp = 0.0;
				for (k = 0; k < K; k++)//找到B的值
				{
					wtemp += weight[i*width*K + j*K + rank_ind[k]];
					if (wtemp > T)
					{
						B = k;
						break;
					}
				}
				k = 0;
				match = 0;
				while ((k <= B) && match == 0)//如果某像素符合背景模型中某一单高斯模型，则此像素为前景像素
				{
					if (abs(diff[i*width*K + j*K + rank_ind[k]]) <= lamba*sigma[i*width*K + j*K + rank_ind[k]])
					{
						fore->imageData[i*width + j] = 255;
						match = 1;
					}
					else//否则为背景，同时利用阈值化去除由于光照变化被误认为为前景的像素点。
					{
						fore->imageData[i*width + j] =data[i*width+j]<120? 0:255;//同时进行阈值化操作
					}
					k++;
				}

			}//end j
		}//end i

		 //膨胀腐蚀操作
		Mat mat = cvarrToMat(fore, true);//转换为mat类
		int g_nStructElementSize = 3; //结构元素(内核矩阵)的尺寸 
		Mat element = getStructuringElement(MORPH_RECT,
			Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1),
			Point(g_nStructElementSize, g_nStructElementSize));
		
		//防止膨胀腐蚀操作把人身体的一些部位（如头部和脚部)当做小区域清除掉，先用高斯滤波使这些部分与躯干连接上)
		GaussianBlur(mat, mat, Size(5, 5), 0, 0);

		//使用膨胀腐蚀先开运算，后闭运算
		erode(mat, mat, element);
		dilate(mat, mat, element);
		dilate(mat, mat, element);
		erode(mat, mat, element);

		IplImage *re = &IplImage(mat);
		uchar * Data = (uchar *)re->imageData;
		//二值化
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				Data[i*width + j] = Data[i*width + j] < 255 ? 0 : 255;
			}
		}

		//设置矩形框
		//找到范围
		int Imax = 0, Jmax = 0, Imin = 1000, Jmin = 1000;
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				if (Data[i*width + j] == 0) {
					if (Imax < i)
						Imax = i;
					if (Jmax < j)
						Jmax = j;
					if (Imin > i)
						Imin = i;
					if (Jmin > j)
						Jmin = j;
				}
			}
		}
		if (Imax*Imin*Jmax*Jmin != 0)//去除一开始的框,即坐标中有0的框。 
		{
			cvRectangle(mframe, cvPoint(Jmin, Imin), cvPoint(Jmax, Imax), cvScalar(0, 0, 255), 3, 4, 0);
			Mat Mframe = cvarrToMat(mframe, true);//转换为mat类
			writer.write(Mframe);
		}
		cvShowImage("组合高斯模型算法删除背景后结果", fore);
		cvShowImage("经过形态学处理后结果", re);
		cvShowImage("原视频处理结果", mframe);
		free(rank_ind);
		cvWaitKey(33);
		mframe = cvQueryFrame(capture);
		if (mframe == NULL)
			return -1;
	}//end while
	cvWaitKey();
	return 0;
}