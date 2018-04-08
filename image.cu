/*
Christopher Ginac

Modified by Derrick Leung and Timothy Moy
image.cpp converted to image.cu
*/

#include <stdlib.h>
#include <iostream>
#include "image.h"
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>

const int ntpb = 16; // number of threads per block
using namespace std;

__device__ bool inBounds(int row, int col, int maxRow, int maxCol) {
	if (row >= maxRow || row < 0 || col >= maxCol || col < 0)
		return false;
	//else
	return true;
}

__global__ void rotateKernel(int* oldImage, int* newImage, int rows, int cols, /*float rads*/ float sinRads, float cosRads) {
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;

	int r0 = rows / 2;
	int c0 = cols / 2;
	//float sinRads = sinf(rads);
	//float cosRads = cosf(rads);

	//float sinRads, cosRads;
	//__sincosf(rads, &sinRads, &cosRads);

	/*__shared__ int s[ntpb * ntpb];
	s[r * cols + c] = oldImage[r * cols + c];*/

	if (r < rows && c < cols)
	{
		int r1 = (int)(r0 + ((r - r0) * cosRads) - ((c - c0) * sinRads));
		int c1 = (int)(c0 + ((r - r0) * sinRads) + ((c - c0) * cosRads));

		if (inBounds(r1, c1, rows, cols))
		{
			newImage[r1 * cols + c1] = oldImage[r * cols + c];
			/*s[r1 * cols + c1] = oldImage[r * cols + c];
			__syncthreads();
			if (threadIdx.x == 0) {
			newImage = s;
			}*/
		}
	}
}

// check reports error if any
//
void check(const char* msg, const cudaError_t err) {
	if (err != cudaSuccess)
		std::cerr << "*** " << msg << ":" << cudaGetErrorString(err) << " ***\n";
}

// report system time
//
void reportTime(const char* msg, chrono::steady_clock::duration span) {
	auto ms = chrono::duration_cast<chrono::milliseconds>(span);
	std::cout << msg << " - took - " <<
		ms.count() << " millisecs" << std::endl;
}

Image::Image()
/* Creates an Image 0x0 */
{
	N = 0;
	M = 0;
	Q = 0;

	pixelVal = NULL;
}

Image::Image(int numRows, int numCols, int grayLevels)
/* Creates an Image of numRows x numCols and creates the arrays for it*/
{

	N = numRows;
	M = numCols;
	Q = grayLevels;

	pixelVal = new int[N * M];
	for (int i = 0; i < N * M; i++)
	{
		pixelVal[i] = 0;
	}
}

Image::~Image()
/*destroy image*/
{
	N = 0;
	M = 0;
	Q = 0;

	delete[] pixelVal;
}

Image::Image(const Image& oldImage)
/*copies oldImage into new Image object*/
{
	N = oldImage.N;
	M = oldImage.M;
	Q = oldImage.Q;

	pixelVal = new int[N * M];
	for (int i = 0; i < N * M; i++)
	{
		pixelVal[i] = oldImage.pixelVal[i];
	}
}

void Image::operator=(const Image& oldImage)
/*copies oldImage into whatever you = it to*/
{
	N = oldImage.N;
	M = oldImage.M;
	Q = oldImage.Q;

	pixelVal = new int[N * M];
	for (int i = 0; i < N * M; i++)
	{
		pixelVal[i] = oldImage.pixelVal[i];
	}
}

void Image::setImageInfo(int numRows, int numCols, int maxVal)
/*sets the number of rows, columns and graylevels*/
{
	N = numRows;
	M = numCols;
	Q = maxVal;
}

void Image::getImageInfo(int &numRows, int &numCols, int &maxVal)
/*returns the number of rows, columns and gray levels*/
{
	numRows = N;
	numCols = M;
	maxVal = Q;
}

int Image::getPixelVal(int row, int col)
/*returns the gray value of a specific pixel*/
{
	return pixelVal[row * M + col];
}


void Image::setPixelVal(int row, int col, int value)
/*sets the gray value of a specific pixel*/
{
	pixelVal[row * M + col] = value;
}

bool Image::inBounds(int row, int col)
/*checks to see if a pixel is within the image, returns true or false*/
{
	if (row >= N || row < 0 || col >= M || col < 0)
		return false;
	//else
	return true;
}

void Image::getSubImage(int upperLeftRow, int upperLeftCol, int lowerRightRow,
	int lowerRightCol, Image& oldImage)
	/*Pulls a sub image out of oldImage based on users values, and then stores it
	in oldImage*/
{
	int width, height;

	width = lowerRightCol - upperLeftCol;
	height = lowerRightRow - upperLeftRow;

	Image tempImage(height, width, Q);

	for (int i = upperLeftRow; i < lowerRightRow; i++)
	{
		for (int j = upperLeftCol; j < lowerRightCol; j++)
			tempImage.pixelVal[(i - upperLeftRow) * M + j - upperLeftCol] = oldImage.pixelVal[i * M + j];
	}

	oldImage = tempImage;
}

int Image::meanGray()
/*returns the mean gray levels of the Image*/
{
	int totalGray = 0;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
			totalGray += pixelVal[i * M + j];
	}

	int cells = M * N;

	return (totalGray / cells);
}

void Image::enlargeImage(int value, Image& oldImage)
/*enlarges Image and stores it in tempImage, resizes oldImage and stores the
larger image in oldImage*/
{
	int rows, cols, gray;
	int pixel;
	int enlargeRow, enlargeCol;

	rows = oldImage.N * value;
	cols = oldImage.M * value;
	gray = oldImage.Q;

	Image tempImage(rows, cols, gray);

	for (int i = 0; i < oldImage.N; i++)
	{
		for (int j = 0; j < oldImage.M; j++)
		{
			pixel = oldImage.pixelVal[i * M + j];
			enlargeRow = i * value;
			enlargeCol = j * value;
			for (int c = enlargeRow; c < (enlargeRow + value); c++)
			{
				for (int d = enlargeCol; d < (enlargeCol + value); d++)
				{
					tempImage.pixelVal[c * M + d] = pixel;
				}
			}
		}
	}

	oldImage = tempImage;
}

void Image::shrinkImage(int value, Image& oldImage)
/*Shrinks image as storing it in tempImage, resizes oldImage, and stores it in
oldImage*/
{
	int rows, cols, gray;

	rows = oldImage.N / value;
	cols = oldImage.M / value;
	gray = oldImage.Q;

	Image tempImage(rows, cols, gray);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			tempImage.pixelVal[i * M + j] = oldImage.pixelVal[i * value * M + j * value];
	}
	oldImage = tempImage;
}

void Image::reflectImage(bool flag, Image& oldImage)
/*Reflects the Image based on users input*/
{
	int rows = oldImage.N;
	int cols = oldImage.M;
	Image tempImage(oldImage);
	if (flag == true) //horizontal reflection
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
				tempImage.pixelVal[(rows - (i + 1)) * M + j] = oldImage.pixelVal[i * M + j];
		}
	}
	else //vertical reflection
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
				tempImage.pixelVal[i * M + (cols - (j + 1))] = oldImage.pixelVal[i * M + j];
		}
	}

	oldImage = tempImage;
}

void Image::translateImage(int value, Image& oldImage)
/*translates image down and right based on user value*/
{
	int rows = oldImage.N;
	int cols = oldImage.M;
	int gray = oldImage.Q;
	Image tempImage(N, M, Q);

	for (int i = 0; i < (rows - value); i++)
	{
		for (int j = 0; j < (cols - value); j++)
			tempImage.pixelVal[(i + value) * M + j + value] = oldImage.pixelVal[i * M + j];
	}

	oldImage = tempImage;
}

void Image::rotateImage(int theta, Image& oldImage)
/*based on users input and rotates it around the center of the image.*/
{
	chrono::steady_clock::time_point ts1, te1;
	ts1 = chrono::steady_clock::now();
	int r0, c0;
	int r1, c1;
	int rows, cols;
	rows = oldImage.N;
	cols = oldImage.M;
	Image tempImage(rows, cols, oldImage.Q);
	chrono::steady_clock::time_point ts2, te2;

	float rads = (theta * 3.14159265) / 180.0;
	float cos1 = cos(rads);
	float sin1 = sin(rads);

	// workspace start
	// - calculate number of blocks for n rows assume square image
	int nb = (rows + ntpb - 1) / ntpb;

	// allocate memory for matrices d_a, d_b on the device

	ts2 = chrono::steady_clock::now();
	// - add your allocation code here
	int* d_a;
	check("device a", cudaMalloc((void**)&d_a, rows* cols * sizeof(int)));
	int* d_b;
	check("device b", cudaMalloc((void**)&d_b, rows* cols * sizeof(int)));
	te2 = chrono::steady_clock::now();
	reportTime("Memory Allocation - Run Time:", te2 - ts2);

	// copy h_a and h_b to d_a and d_b (host to device)
	ts2 = chrono::steady_clock::now();
	// - add your copy code here
	check("copy to d_a", cudaMemcpy(d_a, oldImage.pixelVal, rows * cols * sizeof(int), cudaMemcpyHostToDevice));
	//check("copy to d_b", cudaMemcpy(d_b, tempImage.pixelVal, rows * cols * sizeof(int), cudaMemcpyHostToDevice));
	te2 = chrono::steady_clock::now();
	reportTime("Copy oldImage to Device memory - Run Time:", te2 - ts2);

	// launch execution configuration
	// - define your 2D grid of blocks
	dim3 dGrid(nb, nb);
	// - define your 2D block of threads
	dim3 dBlock(ntpb, ntpb);

	// - launch your execution configuration	
	ts2 = chrono::steady_clock::now();
	rotateKernel << <dGrid, dBlock >> >(d_a, d_b, rows, cols, sin1, cos1);
	te2 = chrono::steady_clock::now();
	reportTime("Kernel - Run Time:", te2 - ts2);

	check("launch error: ", cudaPeekAtLastError());
	// - check for launch termination
	// synchronize the device and the host
	check("Synchronize ", cudaDeviceSynchronize());

	ts2 = chrono::steady_clock::now();
	// copy d_b to tempImage (device to host)
	// - add your copy code here
	check("device copy to hc", cudaMemcpy(tempImage.pixelVal, d_b, rows * cols * sizeof(int), cudaMemcpyDeviceToHost));
	te2 = chrono::steady_clock::now();
	reportTime("Copy device image to tempImage - Run Time:", te2 - ts2);

	// deallocate device memory
	// - add your deallocation code here
	cudaFree(d_a);
	cudaFree(d_b);

	// reset the device
	cudaDeviceReset();
	// workspace end

	ts2 = chrono::steady_clock::now();
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (tempImage.pixelVal[i * M + j] == 0)
				tempImage.pixelVal[i * M + j] = tempImage.pixelVal[i * M + j + 1];
		}
	}
	oldImage = tempImage;
	te2 = chrono::steady_clock::now();
	reportTime("Copy modified to overwrite oldImage - Run Time:", te2 - ts2);

	te1 = chrono::steady_clock::now();
	reportTime("Total Run Time:", te1 - ts1);
}

Image Image::operator+(const Image &oldImage)
/*adds images together, half one image, half the other*/
{
	Image tempImage(oldImage);

	int rows, cols;
	rows = oldImage.N;
	cols = oldImage.M;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			tempImage.pixelVal[i * M + j] = (pixelVal[i * M + j] + oldImage.pixelVal[i * M + j]) / 2;
	}

	return tempImage;
}

Image Image::operator-(const Image& oldImage)
/*subtracts images from each other*/
{
	Image tempImage(oldImage);

	int rows, cols;
	rows = oldImage.N;
	cols = oldImage.M;
	int tempGray = 0;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{

			tempGray = abs(pixelVal[i * M + j] - oldImage.pixelVal[i * M + j]);
			if (tempGray < 35)// accounts for sensor flux
				tempGray = 0;
			tempImage.pixelVal[i * M + j] = tempGray;
		}

	}

	return tempImage;
}

void Image::negateImage(Image& oldImage)
/*negates image*/
{
	int rows, cols, gray;
	rows = N;
	cols = M;
	gray = Q;

	Image tempImage(N, M, Q);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			tempImage.pixelVal[i * M + j] = -(pixelVal[i * M + j]) + 255;
	}

	oldImage = tempImage;
}
