#include <stdio.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "Bosel.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SIZE 10

#define db(a) cout << #a << " = " << a << endl
#define db2(a, b) cout << #a << " = " << a << " " << #b << " = " << b << endl

void performCPU(string filename)
{
	std::clock_t start = clock();
	double duration;

	ImgFloat imagen(filename.c_str());
	//ImgFloat imagen("lena30.jpg");
	// depth, numColors, initialize
	ImgFloat xGradient(imagen.width(), imagen.height(), 1, 1, 0);
	ImgFloat yGradient(imagen.width(), imagen.height(), 1, 1, 0);
	ImgFloat gradientA(imagen.width(), imagen.height(), 1, 1, 0);
	ImgFloat gradientB(imagen.width(), imagen.height(), 1, 1, 0);

	//imagen.blur(1.5);

	ImgFloat R = imagen.get_channel(0);

	Bosel b;
	b.convolution(R, b.Gx, xGradient);
	b.convolution(R, b.Gy, yGradient);

	b.mergeA(gradientA, xGradient, yGradient);
	b.mergeB(gradientB, xGradient, yGradient);
	
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	printf("CPU for image %s takes %.2f seconds\n",filename.c_str(), duration);

	(gradientA, gradientB).display("comparación suma ABSs y SQRT");
	//(xGradient, yGradient, gradient).display("Detección de Bordes");
	cout << duration << endl;
	//gradient.display();
}

__device__ void convolution(int coordinate, float* d_arr, float* gradient, int width, int len,int* mask, int* dir, int* pos)
{
	float c = 0;
	for (int ii = 0; ii < 3; ii++)
	{
		for (int jj = 0; jj < 3; jj++)
		{
			int x = coordinate + width * dir[ii * 3 + jj] + pos[jj];
			if (x >= 0 && x < len)
				c += d_arr[x] * mask[ii * 3 + jj];
		}
	}
	gradient[coordinate] = c;
}

__global__ void deviceComputeGradient(float* d_arr, float* gradient, int width, int len,int* mask, int* dir,int* pos) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x * width + y < len)
		convolution(x * width + y, d_arr, gradient, width, len, mask, dir, pos);
}

__global__ void deviceMerge(float* xGradient, float* yGradient, float* target, int width, int len) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int index = x * width + y;
	if (index < len)
		target[index] = abs(xGradient[index]) + abs(yGradient[index]);
}

void performGPU(string filename) 
{
	ImgFloat imagen(filename.c_str());
	
	std::clock_t start = clock();
	double duration;

	
	ImgFloat img_xGradient(imagen.width(), imagen.height(), 1, 1, 0);
	ImgFloat img_yGradient(imagen.width(), imagen.height(), 1, 1, 0);
	ImgFloat result(imagen.width(), imagen.height(), 1, 1, 0);
	
	// depth, numColors, initialize
	int WIDTH = imagen.width();
	int HEIGHT = imagen.height();
	float *arr, *xGradient, *yGradient, *gradient;
	float *d_arr, *d_xGradient, *d_yGradient, *d_gradient;

	arr = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
	xGradient = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
	yGradient = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
	gradient = (float*)malloc(WIDTH * HEIGHT * sizeof(float));

	cudaMalloc((void**)&d_arr, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void**)&d_xGradient, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void**)&d_yGradient, WIDTH * HEIGHT * sizeof(float));
	cudaMalloc((void**)&d_gradient, WIDTH * HEIGHT * sizeof(float));

	for (int i = 0; i < WIDTH; i++)
		for (int j = 0; j < HEIGHT; j++) {
			arr[i * WIDTH + j] = imagen(i, j);
			xGradient[i * WIDTH + j] = 0;
			yGradient[i * WIDTH + j] = 0;
			gradient[i * WIDTH + j] = 0;
		}
 
	cudaMemcpy(d_arr, arr, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_xGradient, xGradient, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_yGradient, yGradient, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gradient, gradient, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 BLOCKS = dim3(1024, 1024);
	dim3 THREADS = dim3(4, 4);

	/*dim3 BLOCKS(2, 2);
	dim3 THREADS(2, 2);*/
	
	int pos[3] = { -1, 0, 1 };
	int dir[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
	int Gx[9] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};
	int Gy[9] = {
		 1, 2, 1,
		0, 0, 0,
		-1, -2, -1
	};
	int* d_pos, *d_dir, *d_Gx, *d_Gy;
	
	cudaMalloc((void**)&d_pos, 3 * sizeof(int));
	cudaMalloc((void**)&d_dir, 9 * sizeof(int));
	cudaMalloc((void**)&d_Gx, 9 * sizeof(int));
	cudaMalloc((void**)&d_Gy, 9 * sizeof(int));

	cudaMemcpy(d_pos, pos, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dir, dir, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Gx, Gx, 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Gy, Gy, 9 * sizeof(float), cudaMemcpyHostToDevice);

	deviceComputeGradient << < BLOCKS, THREADS >> > (d_arr, d_xGradient, WIDTH, WIDTH * HEIGHT, d_Gx, d_dir, d_pos);
	deviceComputeGradient << < BLOCKS, THREADS >> > (d_arr, d_yGradient, WIDTH, WIDTH * HEIGHT, d_Gy, d_dir, d_pos);
	deviceMerge << < BLOCKS, THREADS >> > (d_xGradient, d_yGradient, d_gradient, WIDTH, WIDTH * HEIGHT);
	//deviceComputeGradient << <BLOCKS, THREADS >> >(d_arr, SIZE);

	cudaMemcpy(xGradient, d_xGradient, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(yGradient, d_yGradient, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(gradient, d_gradient, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < HEIGHT; j++) {
			/*if (i < 10 && j < 10)
				cout << gradient[i * WIDTH + j] << "\t";*/
			img_xGradient(i, j) = xGradient[i * WIDTH + j];
			img_yGradient(i, j) = yGradient[i * WIDTH + j];
			result(i, j) = gradient[i * WIDTH + j];
		}
	}
	
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	printf("GPU for image %s takes %.2f seconds\n", filename.c_str(), duration);

	free(arr);
	free(xGradient);

	cudaFree(d_arr);
	cudaFree(d_xGradient);
	cudaFree(d_yGradient);
	cudaFree(d_gradient);

	cudaFree(d_pos);
	cudaFree(d_dir);
	cudaFree(d_Gx);
	cudaFree(d_Gy);

	(img_xGradient, img_yGradient, result).display("HOLA MUNDO CUDA");
}


int main(int argc, char** argv) {
	
	for (int i = 1; i < 2; i++)
	{
		int len = 4;
		int baseSize = 1024;
		for (int j = 1; j <= len; j++)
		{
			ostringstream stream;
			stream << (baseSize * j);
			if (i == 0)
				performCPU(stream.str() + "x" + stream.str() + ".jpg");
			else 
				performGPU(stream.str() + "x" + stream.str() + ".jpg");
		}
	}

	return 0;
}
