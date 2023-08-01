#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/core/matx.hpp>

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <typeinfo>
#include <fstream>
#include <chrono>

# define PI 3.1415926 

using namespace std;
using namespace cv;

void check(std::vector <cv::Mat1f> const &vec) {
    for (int i = 0; i < vec.size(); i++) {
		int rows = vec.at(i).rows;
		int cols = vec.at(i).cols;
		std::cout << "(" << rows << ", " << cols << ")\n";
	}
}

int main() {
	/*====================
	1. Initialisations
	====================*/
	int width = 2448, height = 2048;
	cv::Size imgSize = Size(width, height);

	// Input & Output File Path
	char leftInputDir[50], rightInputDir[50], inputExt[50];
	sprintf(leftInputDir, "../../Input/20220922/cameraL/");
	sprintf(rightInputDir, "../../Input/20220922/cameraR/");
	sprintf(inputExt, "png");

	char leftOutputDir[50], rightOutputDir[50];
	sprintf(leftOutputDir, "../../Output/result/20220922/cameraL/");
	sprintf(rightOutputDir, "../../Output/result/20220922/cameraR/");

	// Input File Name
	char inputFilename[50];
	sprintf(inputFilename, "%s", "IMG");

	// Frequency Used
	int nFreq = 4;
	vector<float> freqs; // freqs = [1.0, 4.0, 16.0, 48.0];
	freqs.push_back(1.0);
	freqs.push_back(4.0);
	freqs.push_back(16.0);
	freqs.push_back(48.0);

	// No. of steps
	vector<int> nSteps;  // nSteps = [3, 3, 3, 4];
	nSteps.push_back(3);
	nSteps.push_back(3);
	nSteps.push_back(3);
	nSteps.push_back(4);

	// Index of images
	vector<int> nImgs;  // nImgs = [0, 3, 6, 9];
	nImgs.push_back(0);
	nImgs.push_back(3);
	nImgs.push_back(6);
	nImgs.push_back(9);

	// Enable nested parallelism 
	// 1 = enable, 0 = disable
	omp_set_nested(1);

	/*
	bool SavingMode: true or false.
		1. Using specific calculation equation for phase shifting
		2. Using multi-processor for CPU level parallel processing
		3. Do not save any outfile
	*/
	bool saveMode = false;

	/*====================
	2. Load images
	====================*/
	Mat leftImg, rightImg;

	// initialize image containers
	std::vector<cv::Mat1f> leftImgVec, rightImgVec;

	// track load time only AFTER all variable initialisations
	auto loadStart = chrono::high_resolution_clock::now();

	// use two threads (left and right images)
#pragma omp parallel num_threads(2) 
	if (omp_get_thread_num() == 0) {
		// left camera
#pragma omp parallel for num_threads(nFreq)
		for (int freq = 0; freq < nFreq; ++freq) {
			for (int step = 0; step < nSteps[freq]; ++step) {
				char leftImgFile[50];
				sprintf(leftImgFile, "%s%s_%d.%s", leftInputDir, inputFilename, step + nImgs[freq], inputExt);

				// read left camera image
				leftImg = imread(leftImgFile, IMREAD_GRAYSCALE);
				Mat1f leftImgF(leftImg);
				leftImgVec.push_back(leftImgF);
			}
		}
	}
	else {
		// right camera
#pragma omp parallel for num_threads(nFreq)
		for (int freq = 0; freq < nFreq; ++freq) {
			for (int step = 0; step < nSteps[freq]; ++step) {
				char rightImgFile[50];
				sprintf(rightImgFile, "%s%s_%d.%s", rightInputDir, inputFilename, step + nImgs[freq], inputExt);

				// read right camera image
				rightImg = imread(rightImgFile, IMREAD_GRAYSCALE);
				Mat1f rightImgF(rightImg);
				rightImgVec.push_back(rightImgF);
			}
		}
	}

	auto loadEnd = chrono::high_resolution_clock::now();
	auto loadElapsed = chrono::duration_cast<chrono::milliseconds>(loadEnd - loadStart);
	printf("Image load time: %dms\n", loadElapsed.count());

    printf("\nLeft image size: \n");
    check(leftImgVec);
    printf("\nRight image size: \n");
    check(rightImgVec);

	/*====================
	3. Phase shifts
	====================
    Mat leftPhase = Mat::zeros(imgSize, CV_32FC1), rightPhase = Mat::zeros(imgSize, CV_32FC1); 

    // initialize phase shift containers
    std::vector<cv::Mat1f> leftPMVec, rightPMVec;


	auto PSStart = chrono::high_resolution_clock::now();

    // do stuff    

	auto PSEnd = chrono::high_resolution_clock::now();
	auto PSElapsed = chrono::duration_cast<chrono::milliseconds>(PSEnd - PSStart);
	printf("Image load time: %dms\n", PSElapsed.count());
    
    */
	
    /*====================
	4. Phase unwrapping
	====================*/

}
