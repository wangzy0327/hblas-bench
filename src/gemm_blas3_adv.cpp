// gemm_blas3_rocblas.cpp
#include <iostream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas.h>

#include <type_traits>

// Type traits for GEMM configuration (C++11 compatible)
template <typename T>
struct GemmTraits;

template <>
struct GemmTraits<double> {
    typedef double input_type;
    typedef double output_type;
    typedef double scalar_type;
    static const rocblas_datatype a_type = rocblas_datatype_f64_r;
    static const rocblas_datatype b_type = rocblas_datatype_f64_r;
    static const rocblas_datatype c_type = rocblas_datatype_f64_r;
    static const rocblas_datatype compute_type = rocblas_datatype_f64_r;
};

template <>
struct GemmTraits<float> {
    typedef float input_type;
    typedef float output_type;
    typedef float scalar_type;
    static const rocblas_datatype a_type = rocblas_datatype_f32_r;
    static const rocblas_datatype b_type = rocblas_datatype_f32_r;
    static const rocblas_datatype c_type = rocblas_datatype_f32_r;
    static const rocblas_datatype compute_type = rocblas_datatype_f32_r;
};

template <>
struct GemmTraits<half> {
    typedef half input_type;
    typedef half output_type;
    typedef float scalar_type; // compute in f32
    static const rocblas_datatype a_type = rocblas_datatype_f16_r;
    static const rocblas_datatype b_type = rocblas_datatype_f16_r;
    static const rocblas_datatype c_type = rocblas_datatype_f16_r;
    static const rocblas_datatype compute_type = rocblas_datatype_f32_r;
};

template <>
struct GemmTraits<int8_t> {
    typedef int8_t input_type;
    typedef int32_t output_type;   // crucial!
    typedef int32_t scalar_type;   // must be int32 for integer GEMM
    static const rocblas_datatype a_type = rocblas_datatype_i8_r;
    static const rocblas_datatype b_type = rocblas_datatype_i8_r;
    static const rocblas_datatype c_type = rocblas_datatype_i32_r;
    static const rocblas_datatype compute_type = rocblas_datatype_i32_r;
};

// Helper macros for error checking (like CUBLAS_CALL in matmul.cpp)
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

    // Generate arbitrary elements.
    // Generate arbitrary elements in [-8, 7] to fit int8 safely
    int const k = 16807;
    int const m = 16;
    // float value = float(((offset + seed) * k % m) - m / 2);
    int value = ((offset + seed) * k % m) - m / 2; // range: [-8, 7]

    if (std::is_same<T, float>::value) {
      matrix[offset] = static_cast<float>(value);
    }
    else if (std::is_same<T, half>::value) {
      matrix[offset] = __float2half(value);
    }else if (std::is_same<T, int8_t>::value) {
      matrix[offset] = static_cast<int8_t>(value);
    }else if (std::is_same<T, int32_t>::value) {
      matrix[offset] = static_cast<int32_t>(value);
    }else if (std::is_same<T, double>::value) {
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
  HIP_CALL(hipGetLastError());
  HIP_CALL(hipDeviceSynchronize());
}

template <typename T, typename std::enable_if<
            std::is_same<T, double>::value ||
            std::is_same<T, half>::value || std::is_same<T, float>::value
            || std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value,
            bool>::type = true>
void AllocateMatrix(T **matrix, int rows, int columns, int seed = 0) {
  size_t sizeof_matrix = sizeof(T) * rows * columns;

  HIP_CALL(hipMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix));
  HIP_CALL(hipMemset(*matrix, 0, sizeof_matrix));
  InitializeMatrix(*matrix, rows, columns, seed);
}

template <typename InputType,
typename = typename std::enable_if<
    std::is_same<InputType, double>::value ||
    std::is_same<InputType, float>::value ||
    std::is_same<InputType, half>::value ||
    std::is_same<InputType, int8_t>::value>::type>
void TestRocBlasGemm(int M, int N, int K, float alpha_float, float beta_float) {

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

  HIP_CALL(hipMemcpy(C_reference, C, sizeof_C, hipMemcpyDeviceToDevice));

  rocblas_handle handle;
  ROCBLAS_CALL(rocblas_create_handle(&handle));

  // Convert scalars to correct type
  ScalarType alpha = static_cast<ScalarType>(alpha_float);
  ScalarType beta  = static_cast<ScalarType>(beta_float);

  // Warm-up
  std::cout << "Running warm-up..." << std::endl;
  constexpr int warmup_iter = 5;
  for (int i = 0; i < warmup_iter; ++i) {
    ROCBLAS_CALL(rocblas_gemm_ex(handle,
        rocblas_operation_none, rocblas_operation_none,
        M, N, K,
        &alpha,
        A, GemmTraits<InputType>::a_type, lda,
        B, GemmTraits<InputType>::b_type, ldb,
        &beta,
        C, GemmTraits<InputType>::c_type, ldc,
        C, GemmTraits<InputType>::c_type, ldc,  // D == C
        GemmTraits<InputType>::compute_type,
        rocblas_gemm_algo_standard, 0, 0));
  }
  HIP_CALL(hipDeviceSynchronize());

  hipEvent_t start_blas, stop_blas;
  HIP_CALL(hipEventCreate(&start_blas));
  HIP_CALL(hipEventCreate(&stop_blas));
  HIP_CALL(hipEventRecord(start_blas));

  ROCBLAS_CALL(rocblas_gemm_ex(handle,
      rocblas_operation_none, rocblas_operation_none,
      M, N, K,
      &alpha,
      A, GemmTraits<InputType>::a_type, lda,
      B, GemmTraits<InputType>::b_type, ldb,
      &beta,
      C, GemmTraits<InputType>::c_type, ldc,
      C, GemmTraits<InputType>::c_type, ldc,  // D == C
      GemmTraits<InputType>::compute_type,
      rocblas_gemm_algo_standard, 0, 0));

  HIP_CALL(hipEventRecord(stop_blas));
  HIP_CALL(hipEventSynchronize(stop_blas));
  float elapsed_ms_blas;
  HIP_CALL(hipEventElapsedTime(&elapsed_ms_blas, start_blas, stop_blas));
  std::cout << "Rocblas GEMM time: " << elapsed_ms_blas << " ms" << std::endl;

  double flops_blas = 2.0 * M * N * K;
  double gflops_blas = flops_blas / (elapsed_ms_blas * 1e6);
  std::cout << "Rocblas GEMM Performance: " << gflops_blas << " GFLOPS" << std::endl;

  ROCBLAS_CALL(rocblas_destroy_handle(handle));

  // Cleanup
  HIP_CALL(hipFree(C_reference));
  HIP_CALL(hipFree(C));
  HIP_CALL(hipFree(B));
  HIP_CALL(hipFree(A));
}

int main(int argc, char** argv) {
    int problem[3] = {128, 128, 128};
    for (int i = 1; i < argc && i < 4; ++i) {
        std::stringstream ss(argv[i]);
        ss >> problem[i - 1];
    }

    float scalars[2] = {1.0f, 0.0f};
    // 默认设备为 0，可通过 argv[6] 指定
    int device_id = 0;
    if (argc > 6) {
        std::stringstream ss(argv[6]);
        ss >> device_id;
    }
    for (int i = 4; i < argc && i < 6; ++i) {
        std::stringstream ss(argv[i]);
        ss >> scalars[i - 4];
    }

    // int device_id;
    // HIP_CALL(hipGetDevice(&device_id));

    // 设置 HIP 使用指定设备
    HIP_CALL(hipSetDevice(device_id));
    hipDeviceProp_t deviceProp;
    HIP_CALL(hipGetDeviceProperties(&deviceProp, device_id));

    std::cout << "Detected GPU #" << device_id << ": " << deviceProp.name
              << " (Arch: " << deviceProp.gcnArchName << ")" << std::endl;

    // Run test (change to <half> if needed and supported)
    // TestRocBlasGemm<half>(
    // TestRocBlasGemm<int8_t>(
    TestRocBlasGemm<double>(
    // TestRocBlasGemm<float>(
        problem[0], problem[1], problem[2],
        scalars[0], scalars[1]
    );

    std::cout << "Passed." << std::endl;
    return 0;
}