#include <iostream>
#include <stdio.h>
#include <sys/time.h>

using namespace std;

static const int blockSize = 1024;
static const int N = 1000000;

double vectorSum_cpu(float* A, float* sum){
	float lsum = 0;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for(int i = 0; i < N; i++)
      lsum += A[i];
    gettimeofday(&end, NULL);
	*sum = lsum;
    return (end.tv_sec - start.tv_sec) + ((double)end.tv_usec - start.tv_usec)/1000000;
}

__global__ void vs(float* A, float* sum)
{
    int x = threadIdx.x;
	
	if(x >= N) return;
	
	float lsum = 0;
	for(int i = x; i < N; i += blockSize)
		lsum += A[i];
	
	__shared__ float r[blockSize];
	r[x] = lsum;
	
	__syncthreads();
	
	for (int i = blockSize/2; i > 0; i /= 2) {
        if (x < i)
            r[x] += r[x + i];
        __syncthreads();
    }
	
    if (x == 0)
        *sum = r[0];
}

double vectorSum_gpu(float* A, float* sum){
    dim3 blockDim(blockSize);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

	float* cudaA;
	cudaMalloc(&cudaA, N * sizeof(float));
	cudaMemcpy(cudaA, A, N * sizeof(float), cudaMemcpyHostToDevice);

	float* cudaSum;
	cudaMalloc(&cudaSum, sizeof(float));

    float gpu_time = 0.0f;
	
    cudaEventRecord(start);

    vs <<< 1, blockDim >>> (cudaA, cudaSum);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
	
    cudaEventElapsedTime(&gpu_time, start, end);
	
	cudaMemcpy(sum, cudaSum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(cudaSum);
	cudaFree(cudaA);

    return gpu_time/1000;
}

int main() {

    float* A = (float *)malloc(N * sizeof(float));
    float sum_cpu = 0;
    float sum_gpu = 0;
	
    srand(time(0));
    for (int i = 0; i < N; i++)
      A[i] = rand()%10 + 0.1;

    double rez_cpu = vectorSum_cpu(A, &sum_cpu);
    double rez_gpu = vectorSum_gpu(A, &sum_gpu);

	cout << "blockSize = " << blockSize << endl;
    cout << "N = " << N << endl;
    cout << "cpu: " << rez_cpu << " sec" << endl;
    cout << "gpu: " << rez_gpu << " sec" << endl;
    cout << "dif: " << sum_cpu - sum_gpu << endl;
	
    free(A);
    return 0;
}
