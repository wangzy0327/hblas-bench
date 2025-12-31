#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>  // 添加half类型支持
#include <hipblas/hipblas.h>

template<typename T>
__global__ void InitMatrix(T* matrix, int rows, int cols, int seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < rows && j < cols) {
        int k = 16807, m = 16;
        float val = float(((i + j*rows + seed) * k % m) - m/2);
        int offset = i + j*rows;
        
        if constexpr (std::is_same_v<T, hipblasHalf> || std::is_same_v<T, __half>) {
            matrix[offset] = __float2half(val);
        } else {
            matrix[offset] = static_cast<T>(val);
        }
    }
}

template<typename T>
hipError_t AllocMatrix(T** mat, int rows, int cols, int seed) {
    size_t size = sizeof(T) * rows * cols;
    hipError_t err = hipMalloc((void**)mat, size);
    if (err != hipSuccess) return err;
    
    err = hipMemset(*mat, 0, size);
    if (err != hipSuccess) { 
        hipFree((void*)*mat); 
        return err; 
    }
    
    dim3 block(16,16), grid((rows+15)/16, (cols+15)/16);
    InitMatrix<T><<<grid, block>>>(*mat, rows, cols, seed);
    return hipGetLastError();
}

template<typename T>
hipError_t TestGEMM(int M, int N, int K, float alpha, float beta) {
    hipError_t err;
    
    // 分配A, B矩阵
    T *A = nullptr, *B = nullptr;
    float *C = nullptr, *C_ref = nullptr;
    
    // 分配A矩阵 (M×K)
    err = AllocMatrix<T>(&A, M, K, 0);
    if(err != hipSuccess) return err;
    
    // 分配B矩阵 (K×N)
    err = AllocMatrix<T>(&B, K, N, 17);
    if(err != hipSuccess) {
        hipFree((void*)A);
        return err;
    }
    
    // 分配C矩阵 (M×N) - 使用float类型
    err = AllocMatrix<float>(&C, M, N, 101);
    if(err != hipSuccess) {
        hipFree((void*)A);
        hipFree((void*)B);
        return err;
    }
    
    // 分配参考矩阵
    err = AllocMatrix<float>(&C_ref, M, N, 101);
    if(err != hipSuccess) {
        hipFree((void*)A);
        hipFree((void*)B);
        hipFree((void*)C);
        return err;
    }
    
    hipblasHandle_t handle;
    hipblasStatus_t status = hipblasCreate(&handle);
    if(status != HIPBLAS_STATUS_SUCCESS) {
        hipFree((void*)A); 
        hipFree((void*)B); 
        hipFree((void*)C); 
        hipFree((void*)C_ref);
        return hipErrorUnknown;
    }
    
    // 设置数据类型
    hipblasDatatype_t aType, bType, cType, computeType;
    if constexpr (std::is_same_v<T, hipblasHalf> || std::is_same_v<T, __half>) {
        aType = HIPBLAS_R_16F;
        bType = HIPBLAS_R_16F;
        cType = HIPBLAS_R_32F;
        computeType = HIPBLAS_R_32F;
    } else {
        aType = HIPBLAS_R_32F;
        bType = HIPBLAS_R_32F;
        cType = HIPBLAS_R_32F;
        computeType = HIPBLAS_R_32F;
    }
    
    // Warm-up
    for(int i=0; i<5; ++i) {
        status = hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                     M, N, K, &alpha, 
                     (void*)A, aType, M,
                     (void*)B, bType, K, 
                     &beta, 
                     (void*)C, cType, M,
                     computeType, HIPBLAS_GEMM_DEFAULT);
        if(status != HIPBLAS_STATUS_SUCCESS) break;
    }
    hipDeviceSynchronize();
    
    // 计时
    hipEvent_t start, stop;
    hipEventCreate(&start); 
    hipEventCreate(&stop);
    hipEventRecord(start);
    
    status = hipblasGemmEx(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                          M, N, K, &alpha, 
                          (void*)A, aType, M,
                          (void*)B, bType, K, 
                          &beta, 
                          (void*)C, cType, M,
                          computeType, HIPBLAS_GEMM_DEFAULT);
    
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float time_ms = 0.0f;
    hipEventElapsedTime(&time_ms, start, stop);
    
    if(status == HIPBLAS_STATUS_SUCCESS) {
        double gflops = 2.0 * M * N * K / (time_ms * 1e6);
        std::cout << "Time: " << time_ms << " ms, Perf: " << gflops << " GFLOPS\n";
    }
    
    hipblasDestroy(handle);
    hipFree((void*)A); 
    hipFree((void*)B); 
    hipFree((void*)C); 
    hipFree((void*)C_ref);
    
    return (status == HIPBLAS_STATUS_SUCCESS) ? hipSuccess : hipErrorUnknown;
}

int main(int argc, char** argv) {
    int M=128, N=128, K=128;
    float alpha=1.0f, beta=0.0f;
    
    // 解析命令行参数
    if(argc > 1) M = atoi(argv[1]);
    if(argc > 2) N = atoi(argv[2]);
    if(argc > 3) K = atoi(argv[3]);
    if(argc > 4) alpha = atof(argv[4]);
    if(argc > 5) beta = atof(argv[5]);
    
    // 获取设备信息
    hipDeviceProp_t prop;
    int deviceId = 0;
    hipGetDevice(&deviceId);
    hipGetDeviceProperties(&prop, deviceId);
    std::cout << "GPU: " << prop.name << " (" << prop.gcnArchName << ")\n";
    std::cout << "Testing GEMM: M=" << M << ", N=" << N << ", K=" << K 
              << ", alpha=" << alpha << ", beta=" << beta << std::endl;
    
    // 测试float类型
    hipError_t err = TestGEMM<float>(M, N, K, alpha, beta);
    
    if(err != hipSuccess) {
        std::cerr << "Failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    
    std::cout << "Passed.\n";
    return 0;
}
