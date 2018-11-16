//本程序首先利用从摄像头检测到的人脸图片，先进行直方图均衡化
//并缩放到92 * 112的图片大小，然后根据train.txt的采集到的人脸模版
//进行匹配识别（最好是在统一光照下，采集不同角度的人脸图片各一张）
//注意：影响的极大因素在于光照，模版若与采集的图像光照不一样，识别率很低。
//经测试，模板若与检测的图像在同一光照下的话，侧脸，仰脸，正脸均可识别，且识别率较高

#include <stdio.h>  
#include <string.h>  
#include "cv.h"  
#include "cvaux.h"  
#include "highgui.h"  
#include <stdlib.h>  
#include <assert.h>    
#include <math.h>    
#include <float.h>    
#include <limits.h>    
#include <time.h>    
#include <ctype.h>    

////定义几个重要的全局变量  
IplImage ** faceImgArr = 0; // 指向训练人脸和测试人脸的指针（在学习和识别阶段指向不同）  
CvMat    *  personNumTruthMat = 0; // 人脸图像的ID号  
int nTrainFaces = 0; // 训练图像的数目  
int nEigens = 0; // 自己取的主要特征值数目  
IplImage * pAvgTrainImg = 0; // 训练人脸数据的平均值  
IplImage ** eigenVectArr = 0; // 投影矩阵，也即主特征向量  
CvMat * eigenValMat = 0; // 特征值  
CvMat * projectedTrainFaceMat = 0; // 训练图像的投影  
char *filename[5] = { "face1.jpg","face2.jpg","face3.jpg","face4.jpg","face5.jpg" };

static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;
int j = 0;//统计记录的人脸数  
char a[512] = { 0 };
int a1, a2, a3, a4;
time_t timeBegin, timeEnd;
int timeuse;
//// 函数原型  
void learn();
void doPCA();
void storeTrainingData();
int  loadTrainingData(CvMat ** pTrainPersonNumMat);
int  findNearestNeighbor(float * projectedTestFace);
int  loadFaceImgArray(char * filename);
void printUsage();
int detect_and_draw(IplImage* image);
int recognize(IplImage *faceimg);

//主函数，主要包括学习和识别两个阶段，需要运行两次，通过命令行传入的参数区分  
int main(int argc, char** argv)
{
	CvCapture* capture = 0;
	IplImage *frame, *frame_copy = 0;
	int optlen = strlen("--cascade=");
	char *cascade_name = "haarcascade_frontalface_alt2.xml";
	//opencv装好后haarcascade_frontalface_alt2.xml的路径,    
	//也可以把这个文件拷到你的工程文件夹下然后不用写路径名cascade_name= "haarcascade_frontalface_alt2.xml";      
	//或者cascade_name ="C:\\Program Files\\OpenCV\\data\\haarcascades\\haarcascade_frontalface_alt2.xml"    
	cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);

	if (!cascade)
	{
		fprintf(stderr, "ERROR: Could not load classifier cascade\n");
		fprintf(stderr,
			"Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n");
		return -1;
	}
	storage = cvCreateMemStorage(0);
	capture = cvCreateCameraCapture(-1);
	cvNamedWindow("result", 1);

	if (capture)
	{
		timeBegin = time(NULL);
		learn();
		for (;;)
		{
			timeEnd = time(NULL);
			timeuse = timeEnd - timeBegin;//计算经过的时间,统计人数  
			if (!cvGrabFrame(capture))
				break;
			frame = cvRetrieveFrame(capture);
			if (!frame)
				break;
			if (!frame_copy)
				frame_copy = cvCreateImage(cvSize(frame->width, frame->height),
					IPL_DEPTH_8U, frame->nChannels);
			if (frame->origin == IPL_ORIGIN_TL)//如果图像的起点在左上角    
				cvCopy(frame, frame_copy, 0);
			else
				cvFlip(frame, frame_copy, 0);//如果图像的起点不在左上角，而在左下角时，进行X轴对称    

			detect_and_draw(frame_copy);  //检测并且识别  

			if (cvWaitKey(10) >= 0)
				break;
		}

		cvReleaseImage(&frame_copy);
		cvReleaseCapture(&capture);
	}
	else
	{
		printf("Cannot read from CAM");
		return -1;
	}

	cvDestroyWindow("result");

	return 0;
}

//学习阶段代码  
void learn()
{
	int i, offset;

	//加载训练图像集  
	nTrainFaces = loadFaceImgArray("train.txt");
	if (nTrainFaces < 2)
	{
		fprintf(stderr,
			"Need 2 or more training faces\n"
			"Input file contains only %d\n", nTrainFaces);

		return;
	}

	// 进行主成分分析  
	doPCA();

	//将训练图集投影到子空间中  
	projectedTrainFaceMat = cvCreateMat(nTrainFaces, nEigens, CV_32FC1);
	offset = projectedTrainFaceMat->step / sizeof(float);
	for (i = 0; i<nTrainFaces; i++)
	{
		//int offset = i * nEigens;  
		cvEigenDecomposite(
			faceImgArr[i],
			nEigens,
			eigenVectArr,
			0, 0,
			pAvgTrainImg,
			//projectedTrainFaceMat->data.fl + i*nEigens);  
			projectedTrainFaceMat->data.fl + i*offset);
	}

	//将训练阶段得到的特征值，投影矩阵等数据存为.xml文件，以备测试时使用  
	storeTrainingData();
}


//加载保存过的训练结果  
int loadTrainingData(CvMat ** pTrainPersonNumMat)
{
	CvFileStorage * fileStorage;
	int i;


	fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_READ);
	if (!fileStorage)
	{
		fprintf(stderr, "Can't open facedata.xml\n");
		return 0;
	}

	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces * sizeof(IplImage *));
	for (i = 0; i<nEigens; i++)
	{
		char varname[200];
		sprintf(varname, "eigenVect_%d", i);
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}


	cvReleaseFileStorage(&fileStorage);

	return 1;
}

//存储训练结果  
void storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;
	fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_WRITE);

	//存储特征值，投影矩阵，平均矩阵等训练结果  
	cvWriteInt(fileStorage, "nEigens", nEigens);
	cvWriteInt(fileStorage, "nTrainFaces", nTrainFaces);
	cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0, 0));
	for (i = 0; i<nEigens; i++)
	{
		char varname[200];
		sprintf(varname, "eigenVect_%d", i);
		cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0, 0));
	}
	cvReleaseFileStorage(&fileStorage);
}

//寻找最接近的图像  
int findNearestNeighbor(float * projectedTestFace)
{

	double leastDistSq = DBL_MAX;       //定义最小距离，并初始化为无穷大  
	int i, iTrain, iNearest = 0;

	for (iTrain = 0; iTrain<nTrainFaces; iTrain++)
	{
		double distSq = 0;

		for (i = 0; i<nEigens; i++)
		{
			float d_i =
				projectedTestFace[i] -
				projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
			distSq += d_i*d_i / eigenValMat->data.fl[i];  // Mahalanobis算法计算的距离，差的距离的平方除以平均脸的特征值  
														  //  distSq += d_i*d_i; // Euclidean算法计算的距离  
		}

		if (distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}
	//printf("leastdistsq==%f",leastDistSq);  
	return iNearest;
}



//主成分分析  
void doPCA()
{
	int i;
	CvTermCriteria calcLimit;
	CvSize faceImgSize;

	// 自己设置主特征值个数  
	nEigens = nTrainFaces - 1;

	//分配特征向量存储空间  
	faceImgSize.width = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);    //分配个数为住特征值个数  
	for (i = 0; i<nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	//分配主特征值存储空间  
	eigenValMat = cvCreateMat(1, nEigens, CV_32FC1);

	// 分配平均图像存储空间  
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// 设定PCA分析结束条件  
	calcLimit = cvTermCriteria(CV_TERMCRIT_ITER, nEigens, 1);//最大迭代次数为nEigens  

															 // 计算平均图像，特征值，特征向量  
	cvCalcEigenObjects(
		nTrainFaces,
		(void*)faceImgArr,
		(void*)eigenVectArr,
		CV_EIGOBJ_NO_CALLBACK,
		0,
		0,
		&calcLimit,
		pAvgTrainImg,
		eigenValMat->data.fl//存储求得的eigenvalue  
	);

	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}



//加载txt文件的列举的图像  
int loadFaceImgArray(char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int iFace, nFaces = 0;


	if (!(imgListFile = fopen(filename, "r")))
	{
		fprintf(stderr, "Can\'t open file %s\n", filename);
		return 0;
	}

	// 统计人脸数  
	while (fgets(imgFilename, 512, imgListFile)) ++nFaces;//char *fgets(char *buf, int bufsize, FILE *stream);从文件结构体指针stream中读取数据，每次读取一行。读取的数据保存在buf指向的字符数组中，每次最多读取bufsize-1个字符（第bufsize个字符赋'\0'），如果文件中的该行，不足bufsize个字符，则读完该行就结束。如果函数读取成功，则返回指针buf，失败则返回NULL。  
	rewind(imgListFile);//将文件内部的位置指针重新指向一个流（数据流/文件）的开头  

						// 分配人脸图像存储空间和人脸ID号存储空间  
	faceImgArr = (IplImage **)cvAlloc(nFaces * sizeof(IplImage *));
	personNumTruthMat = cvCreateMat(1, nFaces, CV_32SC1);//CvMat* cvCreateMat( int rows, int cols, int type );  

	for (iFace = 0; iFace<nFaces; iFace++)
	{
		// 从文件中读取序号和人脸名称  
		fscanf(imgListFile,
			"%d %s", personNumTruthMat->data.i + iFace, imgFilename);// fscanf(FILE *stream, char *format,[argument...])功 能: 从一个流中执行格式化输入,fscanf遇到空格和换行时结束  

																	 // 加载人脸图像  
		faceImgArr[iFace] = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);

		if (!faceImgArr[iFace])
		{
			fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
			return 0;
		}
	}

	fclose(imgListFile);

	return nFaces;
}



//  
void printUsage()
{
	printf("Usage: eigenface <command>\n",
		"  Valid commands are\n"
		"    train\n"
		"    test\n");
}

int detect_and_draw(IplImage* img)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_VECTOR0, 1, 1, 0, 1, 8);
	static CvScalar colors[] =
	{
		{ { 0,0,255 } },
		{ { 0,128,255 } },
		{ { 0,255,255 } },
		{ { 0,255,0 } },
		{ { 255,128,0 } },
		{ { 255,255,0 } },
		{ { 255,0,0 } },
		{ { 255,0,255 } }
	};

	double scale = 1.3;
	IplImage* gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);
	IplImage* small_img = cvCreateImage(cvSize(cvRound(img->width / scale),
		cvRound(img->height / scale)),
		8, 1);
	int i, personnum = 0;

	cvCvtColor(img, gray, CV_BGR2GRAY);
	cvResize(gray, small_img, CV_INTER_LINEAR);
	cvEqualizeHist(small_img, small_img);
	cvClearMemStorage(storage);

	if (cascade)
	{
		double t = (double)cvGetTickCount();
		CvSeq* faces = cvHaarDetectObjects(small_img, cascade, storage,
			1.1, 2, 0/*CV_HAAR_DO_CANNY_PRUNING*/,
			cvSize(30, 30));
		t = (double)cvGetTickCount() - t;
		//  printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );    
		IplImage* temp_img = cvCreateImage(cvSize(92, 112), 8, 1);
		for (i = 0; i < (faces ? faces->total : 0); i++)
		{
			CvRect* r = (CvRect*)cvGetSeqElem(faces, i);

			IplImage *dst = cvCreateImage(cvSize(r->width, r->height), 8, 1);//cvsize只能选取r->width,r->height不能再后面*scale或+100    
			CvPoint p1;
			p1.x = cvRound((r->x)*scale);
			p1.y = cvRound((r->y)*scale);
			CvPoint p2;
			p2.x = cvRound((r->x + r->width)*scale);
			p2.y = cvRound((r->y + r->height)*scale);
			cvRectangle(img, p1, p2, colors[i % 8], 3, 8, 0);
			cvSetImageROI(small_img, *r);
			cvCopy(small_img, dst);
			cvResize(dst, temp_img);
			cvEqualizeHist(temp_img, temp_img);
			cvResetImageROI(small_img);
			cvSaveImage(filename[i], temp_img);
			cvReleaseImage(&dst);
			//开始识别temp_img  

			personnum = recognize(temp_img);
			if (personnum == 1)
				cvPutText(img, "Yanming", cvPoint(20, 20), &font, CV_RGB(255, 255, 255));//将正确识别的人的姓名显示在屏幕上  

		}
	}

	cvShowImage("result", img);
	cvReleaseImage(&gray);
	cvReleaseImage(&small_img);
	return -1;
}

int recognize(IplImage *faceimg)
{
	int i, nTestFaces = 0;         // 测试人脸数  
	CvMat * trainPersonNumMat = 0;  // 训练阶段的人脸数  
	float * projectedTestFace = 0;
	// 加载保存在.xml文件中的训练结果  
	if (!loadTrainingData(&trainPersonNumMat)) return -3;

	projectedTestFace = (float *)cvAlloc(nEigens * sizeof(float));

	int iNearest, nearest;

	//将测试图像投影到子空间中  
	cvEigenDecomposite(
		faceimg,
		nEigens,
		eigenVectArr,
		0, 0,
		pAvgTrainImg,
		projectedTestFace);
	//cvNormalize(projectedTestFace, projectedTestFace, 1, 0, CV_L1, 0);  
	iNearest = findNearestNeighbor(projectedTestFace);
	nearest = trainPersonNumMat->data.i[iNearest];
	printf("nearest = %d", nearest);
	if (timeuse <= 10)
	{

		if ((nearest == 1) | (nearest == 11) | (nearest == 111))//可以更改train.txt中训练图片的编号,这里将侧脸，仰脸，正脸都归为一起  
			a1++;
		if (nearest == 2)
			a2++;
		if (nearest == 3)
			a3++;
		if (nearest == 4)
			a4++;


		if (a1>7)//如果10s中识别的次数为6则认定为a1  
		{
			printf("yanming\n");
			return 1;
		}
		if (a2>6)
		{
			printf("others\n");
		}
		if (a3>6)
		{
			printf("ma\n");
		}
		if (a4>6)
		{
			printf("ba\n");
		}
	}
	else
	{
		timeBegin = time(NULL);
		a1 = 0;
		a2 = 0;
		a3 = 0;
		a4 = 0;
		return 0;
	}
	return -1;
}