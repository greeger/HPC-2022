#include <iostream>
#include <stdio.h>
#include <sys/time.h>

using namespace std;

static const int blockSize = 32;

double vectorSum_cpu(float* A, float &sum, int N){
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for(int i = 0; i < N; i++)
      sum += A[i];
    gettimeofday(&end, NULL);
    return (end.tv_sec - start.tv_sec) + ((double)end.tv_usec - start.tv_usec)/1000000;
}

__global__ void vs(float* A, float &sum, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	int newN = blockSize * gridDim / 2;
	
	__shared__ int r[newN];
    
	__syncthreads();
	
	for (int i = newN; i > 0; i /= 2) {
        if (x < i)
            r[x] += r[x + i];
        __syncthreads();
    }
    if (x == 0)
        sum = r[0];
 
    return;
}

double vectorSum_gpu(float* A, float &sum, int N){
    dim3 blockDim(blockSize);
	
	int newN;
	if (N <= blockSize) newN = blockSize;
	while(N > newN) newN *= 2;
	
    dim3 gridDim(newN/(blockSize*2));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float gpu_time = 0.0f;
	
    cudaEventRecord(start);

    vs <<< gridDim, blockDim >>> (A, sum, N);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
	
    cudaEventElapsedTime(&gpu_time, start, end);

    return gpu_time/1000;
}

int main() {
    int N = 1000;

    float* A = (float *)malloc(N * sizeof(float));
    float sum_cpu = 0;
    float sum_gpu = 0;
	
    srand(time(0));
    for (int i = 0; i < N; i++)
      A[i] = rand()%10 + 0.1;

    double rez_cpu = vectorSum_cpu(A, sum_cpu, N);
    double rez_gpu = vectorSum_gpu(A, sum_gpu, N);

    cout << "N = " << N << endl;
    cout << "cpu: " << rez_cpu << " sec" << endl;
    cout << "gpu: " << rez_gpu << " sec" << endl;
    cout << "dif: " << sum_cpu - sum_gpu << endl;
	
    free(A);
    return 0;
}
