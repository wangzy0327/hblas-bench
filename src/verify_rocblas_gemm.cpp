#include <stdio.h>
#include <iostream>
#include <string.h>
#include <random>
#include "hip/hip_runtime.h"
#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>
#include "rocblas.h"


#define ROCBLAS_CALL(func) do { \
    rocblas_status status = (func); \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << #func << " failed with status " << status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define HIP_CALL(func) do { \
    hipError_t err = (func); \
    if (err != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << #func << " failed: " << hipGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void sgemm_CPU(int M, int N, int K, float alpha, const float * MatA, const float * MatB, float beta, float * MatC)
{   
    //行主序
    for(int i = 0;i < M;i++){
        for(int j = 0;j < N;j++){
            float sum = 0.0f;
            for(int k = 0;k < K;k++){
				sum += MatA[i*K+k] * MatB[k*N+j];
            }
            MatC[i*N+j] = alpha * sum + beta * MatC[i*N+j];
        }
    }
}

void initialData(float* ip,int size)
{
  time_t t;
  srand((unsigned )time(&t));
  for(int i=0;i<size;i++)
  {
    ip[i]=(float)(rand()&0xffff)/100000.0f;
    // ip[i]=2.0f;
  }
}

__global__ void transpose_gemm(const float* input, float* output, int width, int height){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int input_idx = y * width + x; // 行主序索引
		int output_idx = x * height + y; // 转置后行主序索引
		output[output_idx] = input[input_idx];
	}
}


int main(int argc,char **argv)
{
	int m, n, k;
	m = 2, n = 4, k = 3;
	if (argc == 4) {
		m = std::atoi(argv[1]);
		n = std::atoi(argv[2]);
		k = std::atoi(argv[3]);
		printf("使用命令行参数: M=%d, N=%d, K=%d\n", m, n, k);
	}
	float MatA[] = {1,2,3,
					4,5,6};
	float MatB[] = {1,2,3,4,
					5,6,7,8,
					9,10,11,12};
	//result = {38,44,50,56
				// 83,98,113,128}
	// float* MatA = (float*)malloc(sizeof(float)*m*k);
	// float* MatB = (float*)malloc(sizeof(float)*k*n);
	float* MatC = (float*)malloc(sizeof(float)*m*n);
	float* MatD = (float*)malloc(sizeof(float)*m*n);
	float* MatC_ref = (float*)malloc(sizeof(float)*m*n);

	// initialData(MatA, m*k);
	// initialData(MatB, k*n);
	// initialData(MatC, m*n);
	// initialData(MatC_ref, m*n);

	float* devA,*devB,*devC,*devD,*devD_trans;

	HIP_CALL(hipMalloc(&devA,sizeof(float)*m*k));
	HIP_CALL(hipMalloc(&devB,sizeof(float)*k*n));
	HIP_CALL(hipMalloc(&devC,sizeof(float)*m*n));
	HIP_CALL(hipMalloc(&devD,sizeof(float)*m*n));
	HIP_CALL(hipMalloc(&devD_trans,sizeof(float)*m*n));

	HIP_CALL(hipMemcpy(devA,MatA,m*k*sizeof(float),hipMemcpyHostToDevice));
	HIP_CALL(hipMemcpy(devB,MatB,k*n*sizeof(float),hipMemcpyHostToDevice));
	// HIP_CALL(hipMemcpy(devC,MatC,m*n*sizeof(float),hipMemcpyHostToDevice));
	// HIP_CALL(hipMemcpy(devD,MatC_ref,m*n*sizeof(float),hipMemcpyHostToDevice));
	// HIP_CALL(hipMemcpy(devD_trans,MatC_ref,m*n*sizeof(float),hipMemcpyHostToDevice));

	sgemm_CPU(m,n,k,1.0f,MatA,MatB,0.0f,MatC_ref);

	rocblas_handle handle;
    ROCBLAS_CALL(rocblas_create_handle(&handle));
    float alpha = 1.0f;
    float beta = 0.0f;
    ROCBLAS_CALL(rocblas_gemm_ex(handle,
        rocblas_operation_none, rocblas_operation_none,
        n, m, k,
        &alpha,
        devB, rocblas_datatype_f32_r, n,  //lda = n
        devA, rocblas_datatype_f32_r, k,   //ldb = k
        &beta,
        devC, rocblas_datatype_f32_r, n,
        devC, rocblas_datatype_f32_r, n,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0));
	
	HIP_CALL(hipMemcpy(MatC,devC,m*n*sizeof(float),hipMemcpyDeviceToHost));

	ROCBLAS_CALL(rocblas_gemm_ex(handle,
		rocblas_operation_transpose, rocblas_operation_transpose,
		m, n, k,
		&alpha,
		devA, rocblas_datatype_f32_r, k,  //lda = n
		devB, rocblas_datatype_f32_r, n,   //ldb = k
		&beta,
		devD_trans, rocblas_datatype_f32_r, m,
		devD_trans, rocblas_datatype_f32_r, m,
		rocblas_datatype_f32_r,
		rocblas_gemm_algo_standard, 0, 0));
	
	int blockX = 16, blockY = 16;
	dim3 blockDim(blockX, blockY);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
	transpose_gemm<<<gridDim,blockDim>>>(devD_trans,devD,m,n);

	HIP_CALL(hipMemcpy(MatD,devD,m*n*sizeof(float),hipMemcpyDeviceToHost));

	
	/**
		=== GPU None GEMM 结果 : ===
		38.000000 44.000000 50.000000 56.000000 
		83.000000 98.000000 113.000000 128.000000 

		=== GPU Transpose GEMM 结果 : ===
		38.000000 44.000000 50.000000 56.000000 
		83.000000 98.000000 113.000000 128.000000 

		=== CPU GEMM 结果 : ===
		38.000000 44.000000 50.000000 56.000000 
		83.000000 98.000000 113.000000 128.000000
	*/
	// === 正确性验证===
	printf("\n=== GPU None GEMM 结果 : ===\n");
	float max_error = 0.0f;
	float total_error = 0.0f;
	for(int i = 0;i < m;i++){
		for(int j = 0;j < n;j++){
			int idx = i*n+j;
			float gpu_val = MatC[idx];
			float cpu_val = MatC_ref[idx];
			printf("%.6f ",gpu_val);
		}
		printf("\n");
	}

	printf("\n=== GPU Transpose GEMM 结果 : ===\n");
	for(int i = 0;i < m;i++){
		for(int j = 0;j < n;j++){
			int idx = i*n+j;
			float gpu_val = MatD[idx];
			printf("%.6f ",gpu_val);
		}
		printf("\n");
	}

	printf("\n=== CPU GEMM 结果 : ===\n");
	for(int i = 0;i < m;i++){
		for(int j = 0;j < n;j++){
			int idx = i*n+j;
			float gpu_val = MatC[idx];
			float cpu_val = MatC_ref[idx];
			printf("%.6f ",cpu_val);
		}
		printf("\n");
	}

	HIP_CALL(hipFree(devA));
  	HIP_CALL(hipFree(devB));
  	HIP_CALL(hipFree(devC));
  	HIP_CALL(hipFree(devD));
  	HIP_CALL(hipFree(devD_trans));

	free(MatC);
	free(MatC_ref);
	free(MatD);
	
	return 0;
}

