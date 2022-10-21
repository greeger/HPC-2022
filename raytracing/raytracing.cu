#include <iostream>
#include <string>
#include <stdio.h>
#include <curand.h>
#include <sys/time.h>
#include "EasyBMP.h"

using namespace std;

#define ID(i,j,w) ((j)*(w)+(i))

__global__ void cudaRender(int* cudaPixels, int w, int h, float* spheres, int nSpheres, float* lights, int nLights) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= w || y >= h) return;

    int pos = ID(x, y, w) * 3;
    cudaPixels[pos] = spheres[10];
    cudaPixels[pos + 1] = lights[3];
    cudaPixels[pos + 2] = lights[5];

    return;
}

double render(int nSpheres, int nLights, int w, int h, int* pixels) {

    dim3 blockDim(32, 32);
    dim3 gridDim((w + blockDim.x -1) / blockDim.x, (h + blockDim.y -1) / blockDim.y);

    int* cudaPixels; // r, g, b
    cudaMalloc(&cudaPixels, 3 * w * h * sizeof(int));

    float* cudaSpheres; // x, y, z, R, r, g, b
    cudaMalloc(&cudaSpheres, nSpheres * 7 * sizeof(float));

    float* cudaLights; // x, y, z
    cudaMalloc(&cudaLights, nLights * 6 * sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    struct timeval seed;
    gettimeofday(&seed, NULL);
    curandSetPseudoRandomGeneratorSeed(gen, seed.tv_usec);
    curandGenerateUniform(gen, cudaSpheres, nSpheres * 7);
    curandGenerateUniform(gen, cudaLights, nLights * 3);

    float* spheres = (float*)malloc(nSpheres * 7 * sizeof(float));
    float* lights = (float*)malloc(nLights * 3 * sizeof(float));
    cudaMemcpy(spheres, cudaSpheres, nSpheres * 7 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(lights, cudaLights, nLights * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nSpheres; i++) {
      spheres[i*7 + 0] = spheres[i*7 + 0]*w*2 - w/2;
      spheres[i*7 + 1] = spheres[i*7 + 1]*h*2 - h/2;
      spheres[i*7 + 2] = spheres[i*7 + 2]*w*5 + w;
      spheres[i*7 + 3] = spheres[i*7 + 3]*w/10 + w/20;
    }
    for (int i = 0; i < nLights; i++) {
      lights[i*3 + 0] = lights[i*3 + 0]*w*10 - w*4.5;
      lights[i*3 + 1] = lights[i*3 + 1]*h*10 - h*4.5;
      lights[i*3 + 2] = lights[i*3 + 2]*w*10 - w*2;
    }
    cudaMemcpy(cudaSpheres, spheres, nSpheres * 7 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaLights, lights, nLights * 3 * sizeof(float), cudaMemcpyHostToDevice);
    free(spheres);
    free(lights);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float gpu_time = 0.0f;
    cudaEventRecord(start);

    cudaRender <<< gridDim, blockDim >>> (cudaPixels, w, h, cudaSpheres, nSpheres, cudaLights, nLights);

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
    int w = 1920;
    int h = 1080;
    string fileName = "rez.bmp";

    int* pixels = (int*)malloc(3 * w * h * sizeof(int));

    double rezTime = render(nSpheres, nLights, w, h, pixels);

    //savePic(w, h, pixels, filename);

    cout << "render time: " << rezTime << endl;

    cout << endl << pixels[3] << " " << pixels[4] << " " << pixels[5] << endl;

    free(pixels);
    return 0;
}
