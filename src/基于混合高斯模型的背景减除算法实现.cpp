#include<highgui.hpp>
#include<core.hpp>
#include<opencv.hpp>
#include<cmath>
#include"cv.h"
using namespace std;
using namespace cv;
int main()
{
	CvCapture*capture = cvCreateFileCapture("F:\\�μ�\\�����Ӿ�\\input.avi");//��ȡ��Ƶ
	IplImage* mframe = cvQueryFrame(capture);//��ȡ��Ƶ�еĵ�һ֡
	IplImage* current;//�����ȡ��ͼƬ�ҶȻ����
	IplImage*fore;//ǰ��
	current = cvCreateImage(cvSize(mframe->width, mframe->height), IPL_DEPTH_8U, 1);
	fore = cvCreateImage(cvSize(mframe->width, mframe->height), IPL_DEPTH_8U, 1);
	cvCvtColor(mframe, current, CV_BGR2GRAY);//�ҶȻ�
	uchar* data = (uchar *)current->imageData;
	uchar* data2 = (uchar *)fore->imageData;

	int height = mframe->height;
	int width = mframe->width;

	VideoWriter writer("F:\\�μ�\\�����Ӿ�\\output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
		Size(width, height), 1);

	int K, B;//BΪ��ⱳ��������,KΪ��˹ģ�͸���
	B = 0;
	K = 4;
	double alpha, Sigma0, Wint;//ѧϰ�ʦ�����ʼ����sigma0,��һ����˹ģ�͵ĳ�ʼȨ��Wint
	alpha=0.01;
	Sigma0=10;
	Wint=0.6;
	double lamba = 2.5;
	double T = 0.7;//��ⱳ��������ֵ
	double *miu = new double[width*height*K];//��ֵ
	double *sigma = new double[width*height*K];//��׼��
	double *weight = new double[width*height*K];//Ȩֵ  
	double *diff = new double[width*height*K];//�������ֵ��ֵ  
	double *rank = new double[K];//��˹��������
	//��ʼ��
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)//�Ե�һ֡ÿ������ 
		{
			for (int k = 0; k < K; ++k) //��ÿһ����˹ģ��
			{
				if (k == 0)//�Ե�һ����˹ģ��,��ʼ����ֵΪ��һ֡ͼ������ֵ��ȨֵΪWint
				{
					miu[i*width*K + j*K + k] = data[i*width + j];
					weight[i*width*K + j*K + k] = Wint;
				}
				else
				{//��������˹ģ��,��ֵΪ0��Ȩ��Ϊ(1-Wint)/(K-1)
					miu[i*width*K + j*K + k] = 0;
					weight[i*width*K + j*K + k] = double((1 - Wint) / (K - 1));
				}
				//���и�˹ģ�ͣ���׼���ΪSigma0
				sigma[i*width*K + j*K + k] = Sigma0;
			}
		}
	}
	//��ʼ�����
	int i = 0;
	int j = 0;
	int k = 0;
	while (1)//��ÿһ֡ͼ��
	{
		int *rank_ind = new int[K];
		cvCvtColor(mframe, current, CV_BGR2GRAY);
		//����ÿһ�����أ��ֱ��������ÿһ������˹ģ�͵ľ�ֵ�Ĳ�ֵ
		for (i = 0; i < height; i++)//����ÿһ������
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

				//�������еĵ���˹ģʽ�����������������һ����˹ģʽ����ƥ�䣻��������ز������κεĵ���˹ģʽ����ƥ��
				for (int k = 0; k < K; k++)
				{
					if (diff[i*width*K + j*K + k] <= lamba*sigma[i*width*K + j*K + k])//�ҵ�����ƥ�����С�ĸ�˹ģ�͵ı��
					{
						match = 1;
						l = k;
						break;
					}
				}
				//���ƥ��
				if (match) {
					for (int k = 0; k < K; k++)//�������еĵ���˹ģʽ������Ȩ�غ͹�һ��
					{
						if (k == l)//��������ƥ�����С��k��ƥ��ĸ�˹ģ�ͣ�Ȩ�ز���
						{
							weight[i*width*K + j*K + k] = weight[i*width*K + j*K + k];
						}
						else//��������˹ģ�ͣ�Ȩ�ر�Ϊԭ����(1-��)��
						{
							weight[i*width*K + j*K + k] = (1 - alpha)*weight[i*width*K + j*K + k];
						}
					}
					//Ȩֵ��һ��
					double Mtemp = 0.0;
					for (k = 0; k < K; k++)
					{
						Mtemp += weight[i*width*K + j*K + k];//����K������˹ģʽȨֵ�ĺ�
					}
					for (k = 0; k < K; k++)
					{
						weight[i*width*K + j*K + k] = weight[i*width*K + j*K + k] / Mtemp;//Ȩֵ��һ����ʹ������Ȩֵ��Ϊ1
					}
					//��ƥ��ĸ�˹ģ�͸��²���
					double Sum = 0.0;
					double p = 0.0;
					for (int k = 0; k < K; k++)
					{
						Sum += weight[i*width*K + j*K + k] * double(1 / sqrt(2 * 3.14))*pow(2.7183, -pow((data[i*width + j] - miu[i*width*K + j*K + k]) / sigma[i*width*K + j*K + k], 2));
					}
					//Sum�������ֵgt;
					p = alpha*Sum;
					miu[i*width*K + j*K + l] = (1 - p)*miu[i*width*K + j*K + l] + p*data[i*width + j];
					sigma[i*width*K + j*K + l] = sqrt(1 - p)*sigma[i*width*K + j*K + l] + sqrt(p)*abs(data[i*width + j] - miu[i*width*K + j*K + l]);

				}
				else//��������е���˹ģʽ����ƥ�䣬��Ѱ����СȨ�غ�����׼��.Ȼ����СȨ���滻Ϊһ���µĸ�˹ģʽ
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
					//�ҵ���СȨ�غ�����׼��
					//��Ȩֵ��С��һ���滻Ϊ�µ�
					miu[i*width*K + j*K + tempnum] = data[i*width + j];
					sigma[i*width*K + j*K + tempnum] = sqrt(2)*two_temp;
					weight[i*width*K + j*K + tempnum] = 0.5*one_temp;
					double NMtemp = 0.0;
					//Ȩֵ��һ��
					for (k = 0; k < K; k++)
					{
						NMtemp += weight[i*width*K + j*K + k];//�����ĸ�����˹ģʽȨֵ�ĺ�
					}
					for (k = 0; k < K; k++)
					{
						weight[i*width*K + j*K + k] = weight[i*width*K + j*K + k] / NMtemp;//Ȩֵ��һ����ʹ������Ȩֵ��Ϊ1
					}
				}

				for (k = 0; k < K; k++)//����ÿ������˹ģʽ����Ҫ��
				{
					rank[k] = weight[i*width*K + j*K + k] / sigma[i*width*K + j*K + k];
					rank_ind[k] = k;
				}
				for (k = 0; k < K; k++)//����Ҫ������
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
				for (k = 0; k < K; k++)//�ҵ�B��ֵ
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
				while ((k <= B) && match == 0)//���ĳ���ط��ϱ���ģ����ĳһ����˹ģ�ͣ��������Ϊǰ������
				{
					if (abs(diff[i*width*K + j*K + rank_ind[k]]) <= lamba*sigma[i*width*K + j*K + rank_ind[k]])
					{
						fore->imageData[i*width + j] = 255;
						match = 1;
					}
					else//����Ϊ������ͬʱ������ֵ��ȥ�����ڹ��ձ仯������ΪΪǰ�������ص㡣
					{
						fore->imageData[i*width + j] =data[i*width+j]<120? 0:255;//ͬʱ������ֵ������
					}
					k++;
				}

			}//end j
		}//end i

		 //���͸�ʴ����
		Mat mat = cvarrToMat(fore, true);//ת��Ϊmat��
		int g_nStructElementSize = 3; //�ṹԪ��(�ں˾���)�ĳߴ� 
		Mat element = getStructuringElement(MORPH_RECT,
			Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1),
			Point(g_nStructElementSize, g_nStructElementSize));
		
		//��ֹ���͸�ʴ�������������һЩ��λ����ͷ���ͽŲ�)����С��������������ø�˹�˲�ʹ��Щ����������������)
		GaussianBlur(mat, mat, Size(5, 5), 0, 0);

		//ʹ�����͸�ʴ�ȿ����㣬�������
		erode(mat, mat, element);
		dilate(mat, mat, element);
		dilate(mat, mat, element);
		erode(mat, mat, element);

		IplImage *re = &IplImage(mat);
		uchar * Data = (uchar *)re->imageData;
		//��ֵ��
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				Data[i*width + j] = Data[i*width + j] < 255 ? 0 : 255;
			}
		}

		//���þ��ο�
		//�ҵ���Χ
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
		if (Imax*Imin*Jmax*Jmin != 0)//ȥ��һ��ʼ�Ŀ�,����������0�Ŀ� 
		{
			cvRectangle(mframe, cvPoint(Jmin, Imin), cvPoint(Jmax, Imax), cvScalar(0, 0, 255), 3, 4, 0);
			Mat Mframe = cvarrToMat(mframe, true);//ת��Ϊmat��
			writer.write(Mframe);
		}
		cvShowImage("��ϸ�˹ģ���㷨ɾ����������", fore);
		cvShowImage("������̬ѧ�������", re);
		cvShowImage("ԭ��Ƶ������", mframe);
		free(rank_ind);
		cvWaitKey(33);
		mframe = cvQueryFrame(capture);
		if (mframe == NULL)
			return -1;
	}//end while
	cvWaitKey();
	return 0;
}