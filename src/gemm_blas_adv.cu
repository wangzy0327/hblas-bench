// gemm_blas3_cublas.cu
#include <iostream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <type_traits>

// Type traits for GEMM configuration (C++11 compatible)
template <typename T>
struct GemmTraits;

// FP64 Traits
template <>
struct GemmTraits<double> {
    typedef double input_type;
    typedef double output_type;
    typedef double scalar_type;
    static const cudaDataType_t a_type = CUDA_R_64F;
    static const cudaDataType_t b_type = CUDA_R_64F;
    static const cudaDataType_t c_type = CUDA_R_64F;
    static const cudaDataType_t compute_type = CUDA_R_64F;
};

// FP32 Traits
template <>
struct GemmTraits<float> {
    typedef float input_type;
    typedef float output_type;
    typedef float scalar_type;
    static const cudaDataType_t a_type = CUDA_R_32F;
    static const cudaDataType_t b_type = CUDA_R_32F;
    static const cudaDataType_t c_type = CUDA_R_32F;
    static const cudaDataType_t compute_type = CUDA_R_32F;
};

// FP16 Traits (Tensor Core Optimized)
template <>
struct GemmTraits<half> {
    typedef half input_type;
    typedef half output_type;
    typedef float scalar_type; // compute in f32 for stability
    static const cudaDataType_t a_type = CUDA_R_16F;
    static const cudaDataType_t b_type = CUDA_R_16F;
    static const cudaDataType_t c_type = CUDA_R_16F;
    static const cudaDataType_t compute_type = CUDA_R_32F; // Mixed Precision: FP16 * FP16 + FP32
};

// INT8 Traits
template <>
struct GemmTraits<int8_t> {
    typedef int8_t input_type;
    typedef int32_t output_type;
    typedef int32_t scalar_type;
    static const cudaDataType_t a_type = CUDA_R_8I;
    static const cudaDataType_t b_type = CUDA_R_8I;
    static const cudaDataType_t c_type = CUDA_R_32I;
    static const cudaDataType_t compute_type = CUDA_R_32I;
};

// Helper macros for error checking
#define CUBLAS_CALL(func) do { \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << #func << " failed with status " << status << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CALL(func) do { \
    cudaError_t err = (func); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << #func << " failed: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel to initialize matrix
template <typename T,
          typename std::enable_if<
            std::is_same<T, double>::value ||
            std::is_same<T, half>::value || std::is_same<T, float>::value
            || std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value,
            bool>::type = true>
__global__ void InitializeMatrix_kernel(
  T *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements in [-8, 7]
    int const k = 16807;
    int const m = 16;
    int value = ((offset + seed) * k % m) - m / 2;

    if (std::is_same<T, float>::value) {
      matrix[offset] = static_cast<float>(value);
    }
    else if (std::is_same<T, half>::value) {
      matrix[offset] = __float2half(static_cast<float>(value));
    }
    else if (std::is_same<T, int8_t>::value) {
      matrix[offset] = static_cast<int8_t>(value);
    }
    else if (std::is_same<T, int32_t>::value) {
      matrix[offset] = static_cast<int32_t>(value);
    }
    else if (std::is_same<T, double>::value) {
      matrix[offset] = static_cast<double>(value);
    }
  }
}

template <typename T,
          typename std::enable_if<
            std::is_same<T, double>::value ||
            std::is_same<T, half>::value || std::is_same<T, float>::value
            || std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value,
            bool>::type = true>
void InitializeMatrix(T *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

template <typename T, typename std::enable_if<
            std::is_same<T, double>::value ||
            std::is_same<T, half>::value || std::is_same<T, float>::value
            || std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value,
            bool>::type = true>
void AllocateMatrix(T **matrix, int rows, int columns, int seed = 0) {
  size_t sizeof_matrix = sizeof(T) * rows * columns;

  CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix));
  CUDA_CALL(cudaMemset(*matrix, 0, sizeof_matrix));
  InitializeMatrix(*matrix, rows, columns, seed);
}

template <typename InputType,
typename = typename std::enable_if<
    std::is_same<InputType, double>::value ||
    std::is_same<InputType, float>::value ||
    std::is_same<InputType, half>::value ||
    std::is_same<InputType, int8_t>::value>::type>
void TestCublasGemm(int M, int N, int K, float alpha_float, float beta_float) {

  typedef typename GemmTraits<InputType>::output_type OutputType;
  typedef typename GemmTraits<InputType>::scalar_type ScalarType;

  int lda = M;
  int ldb = K;
  int ldc = M;
  size_t sizeof_C = sizeof(OutputType) * ldc * N;

  InputType* A = nullptr;
  InputType* B = nullptr;
  OutputType* C = nullptr;
  OutputType* C_reference = nullptr;

  AllocateMatrix(&A, M, K, 0);
  AllocateMatrix(&B, K, N, 17);
  AllocateMatrix(&C, M, N, 101);
  AllocateMatrix(&C_reference, M, N, 101);

  CUDA_CALL(cudaMemcpy(C_reference, C, sizeof_C, cudaMemcpyDeviceToDevice));

  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));

  // --- Tensor Core Configuration ---
  // Enable Tensor Core math mode. This is crucial for performance on FP16.
  // For Ampere and later, you might also consider CUBLAS_TF32_TENSOR_OP_MATH
  if (std::is_same<InputType, half>::value) {
      CUBLAS_CALL(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  }
  else if (std::is_same<InputType, float>::value) {
    CUBLAS_CALL(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  }
  // ---------------------------------

  // Convert scalars to correct type
  // ScalarType alpha = static_cast<ScalarType>(alpha_float);
  // ScalarType beta  = static_cast<ScalarType>(beta_float);

  // Warm-up
  std::cout << "Running warm-up..." << std::endl;
  constexpr int warmup_iter = 5;
  for (int i = 0; i < warmup_iter; ++i) {
    CUBLAS_CALL(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha_float,
        A, GemmTraits<InputType>::a_type, lda,
        B, GemmTraits<InputType>::b_type, ldb,
        &beta_float,
        C, GemmTraits<InputType>::c_type, ldc,
        GemmTraits<InputType>::compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // Use Tensor Core heuristic
  }
  CUDA_CALL(cudaDeviceSynchronize());
  // 定义测试迭代次数
  constexpr int test_iter = 50;
  std::cout << "Running performance test (" << test_iter << " iterations)..." << std::endl;

  cudaEvent_t start_blas, stop_blas;
  CUDA_CALL(cudaEventCreate(&start_blas));
  CUDA_CALL(cudaEventCreate(&stop_blas));
  CUDA_CALL(cudaEventRecord(start_blas));

  // 执行50次 GEMM 操作
  for (int i = 0; i < test_iter; ++i) {
    CUBLAS_CALL(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha_float,
        A, GemmTraits<InputType>::a_type, lda,
        B, GemmTraits<InputType>::b_type, ldb,
        &beta_float,
        C, GemmTraits<InputType>::c_type, ldc,
        GemmTraits<InputType>::compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // Use Tensor Core heuristic
  }

  CUDA_CALL(cudaEventRecord(stop_blas));
  CUDA_CALL(cudaEventSynchronize(stop_blas));
  float total_elapsed_ms_blas;
  CUDA_CALL(cudaEventElapsedTime(&total_elapsed_ms_blas, start_blas, stop_blas));
  float avg_elapsed_ms_blas = total_elapsed_ms_blas / test_iter;
  std::cout << "Cublas Avg GEMM time: " << avg_elapsed_ms_blas << " ms" << std::endl;

  double flops_blas = 2.0 * M * N * K;
  double gflops_blas = flops_blas / (avg_elapsed_ms_blas * 1e6);
  std::cout << "Cublas Avg GEMM Performance: " << gflops_blas << " GFLOPS" << std::endl;

  CUBLAS_CALL(cublasDestroy(handle));

  // Cleanup
  CUDA_CALL(cudaFree(C_reference));
  CUDA_CALL(cudaFree(C));
  CUDA_CALL(cudaFree(B));
  CUDA_CALL(cudaFree(A));
}

int main(int argc, char** argv) {
    int problem[3] = {128, 128, 128};
    for (int i = 1; i < argc && i < 4; ++i) {
        std::stringstream ss(argv[i]);
        ss >> problem[i - 1];
    }

    float scalars[2] = {1.0f, 0.0f};
    int device_id = 0;
    if (argc > 6) {
        std::stringstream ss(argv[6]);
        ss >> device_id;
    }
    for (int i = 4; i < argc && i < 6; ++i) {
        std::stringstream ss(argv[i]);
        ss >> scalars[i - 4];
    }

    CUDA_CALL(cudaSetDevice(device_id));
    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device_id));

    std::cout << "Detected GPU #" << device_id << ": " << deviceProp.name
              << " (Compute Capability: " << deviceProp.major << "." << deviceProp.minor << ")" << std::endl;

    // Run test with Half precision (FP16) to utilize Tensor Cores
    // TestCublasGemm<half>(
    TestCublasGemm<float>(
    // TestCublasGemm<double>(
    // TestCublasGemm<int8_t>(
        problem[0], problem[1], problem[2],
        scalars[0], scalars[1]
    );

    std::cout << "Passed." << std::endl;
    return 0;
}
