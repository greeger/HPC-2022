#include <iostream>
#include <stdio.h>
#include <sys/time.h>

using namespace std;

#define ID(i,j,rows) ((j)*(rows)+(i))

double matmul_cpu(float* A, float* B, float* C, int K, int M, int N){
    struct timeval start, end;
    gettimeofday(&start, NULL);
    float t;
    for(int i = 0; i < K; i++)
      for(int j = 0; j < N; j++){
        t = 0;
        for(int k = 0; k < M; k++)
          t += A[ID(i,k,K)] * B[ID(k,j,M)];
        C[ID(i,j,K)] = t;
      }
    gettimeofday(&end, NULL);
    return (end.tv_sec - start.tv_sec) + ((double)end.tv_usec - start.tv_usec)/1000000;
}

__global__ void MatMul(float* A, float* B, float* C, int K, int M, int N)
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

    cudaDeviceSynchronize();
    cudaEventRecord(end);
	
    cudaEventElapsedTime(&gpu_time, start, end);

    cudaMemcpy(C, cudaC, K * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);

    return gpu_time*1000;
}

void print_matrix(float* A, int rows, int cols){
    for(int i = 0; i < rows; i++){
      for(int j = 0; j < cols; j++)
        cout << A[ID(i,j,rows)] << " ";
      cout << endl;
    }
    cout << endl;
}

float dif_matrix(float* A, float* B, int l){
    float rez = 0;
    for(int i = 0; i < l; i++)
      rez += (A[i] - B[i]) * (A[i] - B[i]);
    return rez;
}

int main() {
    int same = 1000;
	  int K = same, M = same, N = same;

    float* A = (float *)malloc(K * M * sizeof(float));
    float* B = (float *)malloc(M * N * sizeof(float));
    float* C_cpu = (float *)malloc(K * N * sizeof(float));
    float* C_gpu = (float *)malloc(K * N * sizeof(float));
	
    srand(time(0));
    for (int j = 0; j < K*M; j++)
      A[j] = rand()%10 + 0.1;
    for (int j = 0; j < M*N; j++)
      B[j] = rand()%10 + 0.2;

    double rez_cpu = matmul_cpu(A, B, C_cpu, K, M, N);
    double rez_gpu = matmul_gpu(A, B, C_gpu, K, M, N);

    //print_matrix(A, K, M);
    //print_matrix(B, M, N);
    //print_matrix(C_cpu, K, N);

    cout << "N = " << same << endl;
    cout << "cpu: " << rez_cpu << " sec" << endl;
    cout << "gpu: " << rez_gpu << " sec" << endl;
    cout << "dif: " << dif_matrix(C_cpu, C_gpu, K * N) << endl;
	
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
    return 0;
}
