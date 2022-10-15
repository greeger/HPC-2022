#include <iostream>
#include <stdio.h>
#include <sys/time.h>

using namespace std;

double vectorSum_cpu(float* A, float &sum, int N){
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for(int i = 0; i < N; i++)
      sum += A[i];
    gettimeofday(&end, NULL);
    return (end.tv_sec - start.tv_sec) + ((double)end.tv_usec - start.tv_usec)/1000000;
}

/*__global__ void MatMul(float* A, float* B, float* C, int K, int M, int N)
{
    int x = blockIdx.y * blockDim.y + threadIdx.x;
    int y = blockIdx.x * blockDim.x + threadIdx.y;

    if(y >= N || x >= K)
      return;

    float value = 0.0;
    for(int i = 0; i < M; i++)
        value += A[ID(x,i,K)] * B[ID(i,y,M)];
    C[ID(x,y,K)] = value;
 
    return;
}

double matmul_gpu(float* A, float* B, float* C, int K, int M, int N){
    dim3 blockDim(32, 32);
    dim3 gridDim((N-1)/blockDim.x + 1, (K-1)/blockDim.y + 1);

    float* cudaA;
    float* cudaB;
    float* cudaC;

    cudaMalloc(&cudaA, K * M * sizeof(float));
    cudaMalloc(&cudaB, M * N * sizeof(float));
    cudaMalloc(&cudaC, K * N * sizeof(float));

    cudaMemcpy(cudaA, A, K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, B, M * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    float gpu_time = 0.0f;
	
    cudaEventRecord(start);

    MatMul <<< gridDim, blockDim >>> (cudaA, cudaB, cudaC, K, M, N);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
	
    cudaEventElapsedTime(&gpu_time, start, end);

    cudaMemcpy(C, cudaC, K * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);

    return gpu_time/1000;
}*/

int main() {
    int N = 100;

    float* A = (float *)malloc(N * sizeof(float));
    float sum_cpu = 0;
    float sum_gpu = 0;
	
    srand(time(0));
    for (int i = 0; i < N; i++)
      A[i] = rand()%10 + 0.1;

    double rez_cpu = vectorSum_cpu(A, sum_cpu, N);
    double rez_gpu = 0;//vectorSum_gpu(A, sum_gpu, N);

    cout << "N = " << N << endl;
    cout << "cpu: " << rez_cpu << " sec" << endl;
    cout << "gpu: " << rez_gpu << " sec" << endl;
    cout << "dif: " << sum_cpu - sum_gpu << endl;
	
    free(A);
    return 0;
}
