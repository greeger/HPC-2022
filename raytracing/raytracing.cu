#include <iostream>
#include <math.h>
#include <curand.h>
#include <sys/time.h>
#include "EasyBMP_1.06/EasyBMP.h"

using namespace std;

#define ID(i,j,w) ((j)*(w)+(i))
#define INF 100000007

struct Coord {

    float x, y, z;
	
    __device__ __host__ Coord(float xx, float yy, float zz) {
	  x = xx;
	  y = yy;
	  z = zz;
	}
	
    __device__ __host__ Coord() {
	  x = 0;
	  y = 0;
	  z = 0;
	}
};

struct Ray {

    Coord rho;
	Coord e;
	
    __device__ __host__ Ray(Coord rrho, Coord ee) {
	  rho = Coord(rrho.x, rrho.y, rrho.z);
	  e = Coord(ee.x, ee.y, ee.z);
	}
	
    __device__ __host__ Ray() {
	  rho = Coord();
	  e = Coord();
	}
};

struct Sphere {

    Coord position;
	float r;
	Coord color;
	
    __device__ __host__ Sphere(Coord pposition, float rr, Coord ccolor) {
	  position = Coord(pposition.x, pposition.y, pposition.z);
	  r = rr;
	  color = Coord(ccolor.x, ccolor.y, ccolor.z);
	}
};

void saveBMP(Coord* pixels, int w, int h, const char* filename) {
    BMP image;
    image.SetSize(w, h);
    RGBApixel pixel;
    pixel.Alpha = 0;
    for (int i = 0; i < w; i++)
      for (int j = 0; j < h; j++) {
        pixel.Red = pixels[ID(i, j, w)].x;
        pixel.Green = pixels[ID(i, j, w)].y;
        pixel.Blue = pixels[ID(i, j, w)].z;
        image.SetPixel(i, j, pixel);
      }
    image.WriteToFile(filename);
}

__device__ float getNorm(Coord n) {
    return sqrt((n.x*n.x+n.y*n.y+n.z*n.z));
}

__device__ Coord normalize(Coord n){
    float norm = getNorm(n);
    return Coord(n.x/norm, n.y/norm, n.z/norm);
}

__device__ float dotProduct(Coord n, Coord v){
    return n.x*v.x + n.y*v.y + n.z*v.z;
}

__device__ float normDotProduct(Coord n, Coord v){
    return dotProduct(n, v)/(getNorm(n)*getNorm(v));
}

__device__ Coord linearCombination(Coord a, float m, Coord b, float n){
    return Coord(a.x*m + b.x*n, a.y*m + b.y*n, a.z*m + b.z*n);
}

__device__ Coord getRayPoint(Ray ray, float t) {
    return linearCombination(ray.rho, 1, ray.e, t);
}

__device__ float getIntersection(Sphere sphere, Ray ray){
    Coord shift = linearCombination(ray.rho, 1, sphere.position, -1);
    float d = pow(dotProduct(shift, ray.e),2) - dotProduct(shift, shift) + sphere.r*sphere.r;
    if(d < 0) return -INF;
    return -dotProduct(shift, ray.e) - sqrt(d);
}

__device__ Coord getNormal(Coord intersection, Sphere sphere){
    return normalize(linearCombination(intersection, 1, sphere.position, -1));
}

__device__ Ray getReflectedRay(Ray ray, float t, Coord n){
    return Ray(getRayPoint(ray, t), normalize(linearCombination(ray.e, 1, n, -2*dotProduct(ray.e, n))));
}

struct ReturnValue {

    Coord color = Coord();
	Ray ray = Ray();
};

__device__ ReturnValue rayTrace(Ray ray, Sphere* spheres, int nSpheres, Coord* lights, int nLights) {
	int closestSphere = -1;
	float t = INF;
    for (int i = 0; i < nSpheres; i++) {
	  float currT = getIntersection(spheres[i], ray);
	  if(currT > 0 && currT < t && getRayPoint(ray, currT).z > 0) {
	    closestSphere = i;
		t = currT;
	  }
    }
    if (closestSphere >= 0){
      Coord intersection = getRayPoint(ray, t);
      Coord currColor = Coord();
      for (int i = 0; i < nLights; i++) {
        Ray lightRay = Ray(intersection, normalize(linearCombination(lights[i], 1, intersection, -1)));
        bool isIntersection = 0;
        for (int i = 0; i < nSpheres; i++)
          if(getIntersection(spheres[i], lightRay) > 0) isIntersection = 1;
        if(!isIntersection) {
		  float k = normDotProduct(lightRay.e, getNormal(intersection, spheres[closestSphere]))/nLights;
		  if (k > 0)
            currColor = linearCombination(currColor, 1, spheres[closestSphere].color, k);
        }
      }
      Ray reflectedRay = getReflectedRay(ray, t, getNormal(intersection, spheres[closestSphere]));
      ReturnValue value;
      value.ray = reflectedRay;
      value.color = currColor;
      return value;
    }
    return ReturnValue();
}

__global__ void cudaRender(Coord* cudaPixels, int w, int h,
  Sphere* spheres, int nSpheres, Coord* lights, int nLights, Coord cam) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= w || y >= h) return;

    Ray ray = Ray(cam, normalize(linearCombination(Coord(x, y, 0), 1, cam, -1)));
    Coord color = Coord();

    float glossy = 0.2;
    bool isOk = 1;
    for(int i = 0; i < 5; i++){
      if(isOk){
        ReturnValue value = rayTrace(ray, spheres, nSpheres, lights, nLights);
        ray = value.ray;
        if (getNorm(ray.e) < 0.5) isOk = 0;
          color = linearCombination(color, 1, value.color, (1 - glossy)*pow(glossy, i));
      }
    }

    cudaPixels[ID(x, y, w)].x = (int)(color.x*255);
    cudaPixels[ID(x, y, w)].y = (int)(color.y*255);
    cudaPixels[ID(x, y, w)].z = (int)(color.z*255);

    return;
}

double render(int nSpheres, int nLights, int w, int h, Coord* pixels) {

    dim3 blockDim(32, 32);
    dim3 gridDim((w + blockDim.x -1) / blockDim.x, (h + blockDim.y -1) / blockDim.y);

    Sphere* spheres = (Sphere*)malloc(nSpheres * sizeof(Sphere));
    Coord* lights = (Coord*)malloc(nLights * sizeof(Coord));
	
    Coord* cudaPixels;
    cudaMalloc(&cudaPixels, 3 * w * h * sizeof(int));
    Sphere* cudaSpheres;
    cudaMalloc(&cudaSpheres, nSpheres * sizeof(Sphere));
    Coord* cudaLights;
    cudaMalloc(&cudaLights, nLights * sizeof(Coord));

    float* rand = (float*)malloc((nSpheres*7 + nLights*3) * sizeof(float));
    float* cudaRand;
    cudaMalloc(&cudaRand, (nSpheres*7 + nLights*3) * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    struct timeval seed;
    gettimeofday(&seed, NULL);
    curandSetPseudoRandomGeneratorSeed(gen, seed.tv_usec);
    curandGenerateUniform(gen, cudaRand, (nSpheres*7 + nLights*3));
    cudaMemcpy(rand, cudaRand, (nSpheres*7 + nLights*3) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(cudaRand);

    for (int i = 0; i < nSpheres; i++) {
      spheres[i].position.x = rand[i*7 + 0]*w*2 - w/2;
      spheres[i].position.y = rand[i*7 + 1]*h*2 - h/2;
      spheres[i].position.z = rand[i*7 + 2]*w + w;
      spheres[i].r = rand[i*7 + 3]*w/10 + w/6;
	  spheres[i].color.x = rand[i*7 + 4];
	  spheres[i].color.y = rand[i*7 + 5];
	  spheres[i].color.z = rand[i*7 + 6];
    }
    for (int i = 0; i < nLights; i++) {
      lights[i].x = rand[nSpheres*7 + i*3 + 0]*w*10 - w*4.5;
      lights[i].y = rand[nSpheres*7 + i*3 + 1]*h*10 - h*4.5;
      lights[i].z = rand[nSpheres*7 + i*3 + 2]*w*10 - w*2;
    }
    cudaMemcpy(cudaSpheres, spheres, nSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaLights, lights, nLights * sizeof(Coord), cudaMemcpyHostToDevice);
    free(spheres);
    free(lights);
	free(rand);
	
	Coord cam = Coord(w/2, h/2, -2*w);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float gpu_time = 0.0f;
    cudaEventRecord(start);

    cudaRender <<< gridDim, blockDim >>> (cudaPixels, w, h, cudaSpheres, nSpheres, cudaLights, nLights, cam);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_time, start, end);

    cudaMemcpy(pixels, cudaPixels, 3 * w * h * sizeof(int), cudaMemcpyDeviceToHost);
	
    cudaFree(cudaPixels);
    cudaFree(cudaSpheres);
    cudaFree(cudaLights);

    return gpu_time/1000;
}

int main() {

    int nSpheres = 10;
    int nLights = 2;
    int w = 1920;
    int h = 1080;
    const char* fileName = "rez.bmp";

    Coord* pixels = (Coord*)malloc(w * h * sizeof(Coord));

    double rezTime = render(nSpheres, nLights, w, h, pixels);

    saveBMP(pixels, w, h, fileName);

    cout << "render time: " << rezTime << endl;

    free(pixels);
    return 0;
}
