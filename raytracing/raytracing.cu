#include <iostream>
#include <string>
#include <stdio.h>
#include <curand.h>
#include "EasyBMP.h"

using namespace std;

#define ID(i,j,w) ((j)*(w)+(i))

__global__ void cudaRender(int* cudaPixels, int w, int h, float* spheres, int nSpheres, float* lights, int nLights) {
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= w || y >= h) return;
	
    int pos = ID(x, y, w) * 3;
    cudaPixels[pos] = lights[0];
    cudaPixels[pos + 1] = spheres[1];
    cudaPixels[pos + 2] = spheres[nSpheres * 4 - 1];

    return;
}

double render(int nSpheres, int nLights, int w, int h, int* pixels) {
    
    dim3 blockDim(32, 32);
    dim3 gridDim((w + blockDim.x -1) / blockDim.x, (h + blockDim.y -1) / blockDim.y);

    int* cudaPixels;
    cudaMalloc(&cudaPixels, 3 * w * h * sizeof(int));

    float* spheres;
    cudaMalloc(&spheres, nSpheres * 4 * sizeof(float));
    
    float* lights;
    cudaMalloc(&lights, nLights * 6 * sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, spheres, nSpheres * 4);
    curandGenerateUniform(gen, lights, nLights * 3);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float gpu_time = 0.0f;
    cudaEventRecord(start);

    cudaRender <<< gridDim, blockDim >>> (cudaPixels, w, h, spheres, nSpheres, lights, nLights);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_time, start, end);

    cudaMemcpy(pixels, cudaPixels, 3 * w * h * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cudaPixels);

    return gpu_time/1000;
}

int main() {

    int nSpheres = 7;
    int nLights = 2;
    int w = 40;//1920;
    int h = 20;//1080;
    string fileName = "rez.bmp";

    int* pixels = (int*)malloc(3 * w * h * sizeof(int));

    double rezTime = render(nSpheres, nLights, w, h, pixels);

    //savePic(w, h, pixels, filename);

    cout << "render time: " << rezTime << endl;

    cout << endl << endl;
    for(int i = 0; i < w * h; i += 3)
      cout << pixels[i] << " ";
    cout << endl << endl;
    for(int i = 1; i < w * h; i += 3)
      cout << pixels[i] << " ";
    cout << endl << endl;
    for(int i = 2; i < w * h; i += 3)
      cout << pixels[i] << " ";

    free(pixels);
    return 0;
}