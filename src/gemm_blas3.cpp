// gemm_blas.cpp
#include <iostream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>

template <typename T,
          std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
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
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    if constexpr (std::is_same_v<T, float>) {
      matrix[offset] = value;
    }
    else if constexpr (std::is_same_v<T, half>) {
      matrix[offset] = __float2half(value);
    }
  }
}

template <typename T,
          std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
hipError_t InitializeMatrix(T *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

  return hipGetLastError();
}

template <typename T,
          std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
hipError_t AllocateMatrix(T **matrix, int rows, int columns, int seed = 0) {
  hipError_t result;

  size_t sizeof_matrix = sizeof(T) * rows * columns;

  result = hipMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);
  if (result != hipSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << hipGetErrorString(result) << std::endl;
    return result;
  }

  result = hipMemset(*matrix, 0, sizeof_matrix);
  if (result != hipSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << hipGetErrorString(result) << std::endl;
    (void)hipFree(*matrix);  // 注意：这里也应释放，但原逻辑未做；为简化，按原逻辑
    return result;
  }

  result = InitializeMatrix(*matrix, rows, columns, seed);
  if (result != hipSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << hipGetErrorString(result) << std::endl;
    (void)hipFree(*matrix);
    return result;
  }

  return result;
}

template <typename T, std::enable_if_t<
            std::is_same_v<T, half> || std::is_same_v<T, float>,
            bool> = true>
hipError_t TestHipBlasGemm(int M, int N, int K, float alpha, float beta) {
  hipError_t result;
  hipblasStatus_t status;

  int lda = M;
  int ldb = K;
  int ldc = M;
  size_t sizeof_C = sizeof(float) * ldc * N;

  T *A = nullptr;
  T *B = nullptr;
  float *C = nullptr;
  float *C_reference = nullptr;

  result = AllocateMatrix(&A, M, K, 0);
  if (result != hipSuccess) return result;

  result = AllocateMatrix(&B, K, N, 17);
  if (result != hipSuccess) {
    (void)hipFree(A);
    return result;
  }

  result = AllocateMatrix(&C, M, N, 101);
  if (result != hipSuccess) {
    (void)hipFree(A);
    (void)hipFree(B);
    return result;
  }

  result = AllocateMatrix(&C_reference, M, N, 101);
  if (result != hipSuccess) {
    (void)hipFree(A);
    (void)hipFree(B);
    (void)hipFree(C);
    return result;
  }

  result = hipMemcpy(C_reference, C, sizeof_C, hipMemcpyDeviceToDevice);
  if (result != hipSuccess) {
    std::cerr << "Failed to copy C_hipBlas matrix to C_reference: "
      << hipGetErrorString(result) << std::endl;
    (void)hipFree(C_reference);
    (void)hipFree(C);
    (void)hipFree(B);
    (void)hipFree(A);
    return result;
  }

  hipblasHandle_t handle;
  hipblasStatus_t hipblas_status = hipblasCreate(&handle);
  if (hipblas_status != HIPBLAS_STATUS_SUCCESS) {
    std::cerr << "hipBLAS handle creation failed: " << hipblas_status << std::endl;
    (void)hipFree(C_reference);
    (void)hipFree(C);
    (void)hipFree(B);
    (void)hipFree(A);
    return hipErrorUnknown;
  }

  hipblasDatatype_t aType, bType, cType, computeType;
  if constexpr (std::is_same_v<T, half>) {
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
  std::cout << "Running warm-up..." << std::endl;
  constexpr int warmup_iter = 5;
  for (int i = 0; i < warmup_iter; ++i) {
    hipblas_status = hipblasGemmEx(handle,
                              HIPBLAS_OP_N, HIPBLAS_OP_N,
                              M, N, K,
                              &alpha,
                              A, aType, lda,
                              B, bType, ldb,
                              &beta,
                              C, cType, ldc,
                              computeType,
                              HIPBLAS_GEMM_DEFAULT);
    if (hipblas_status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLAS warm-up failed!" << std::endl;
        (void)hipblasDestroy(handle);
        (void)hipFree(C); (void)hipFree(B); (void)hipFree(A);
        (void)hipFree(C_reference);
        return hipErrorUnknown;
    }
  }
  (void)hipDeviceSynchronize();

  hipEvent_t start_blas, stop_blas;
  (void)hipEventCreate(&start_blas);
  (void)hipEventCreate(&stop_blas);
  (void)hipEventRecord(start_blas);

  hipblas_status = hipblasGemmEx(handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        M, N, K,
        &alpha,
        A, aType, lda,
        B, bType, ldb,
        &beta,
        C, cType, ldc,
        computeType,
        HIPBLAS_GEMM_DEFAULT);

  if (hipblas_status != HIPBLAS_STATUS_SUCCESS) {
    std::cerr << "hipBLAS GEMM failed!" << std::endl;
    (void)hipblasDestroy(handle);
    (void)hipFree(C); (void)hipFree(B); (void)hipFree(A);
    (void)hipFree(C_reference);
    return hipErrorUnknown;
  }

  (void)hipEventRecord(stop_blas);
  (void)hipEventSynchronize(stop_blas);
  float elapsed_ms_blas;
  (void)hipEventElapsedTime(&elapsed_ms_blas, start_blas, stop_blas);
  std::cout << "Hipblas GEMM time: " << elapsed_ms_blas << " ms" << std::endl;

  double flops_blas = 2.0 * M * N * K;
  double gflops_blas = flops_blas / (elapsed_ms_blas * 1e6);
  std::cout << "Hipblas GEMM Performance: " << gflops_blas << " GFLOPS" << std::endl;

  (void)hipblasDestroy(handle);

  // Final cleanup
  (void)hipFree(C_reference);
  (void)hipFree(C);
  (void)hipFree(B);
  (void)hipFree(A);

  // IMPORTANT: Ensure non-void function returns a value on all paths!
  return hipSuccess;
}

int main(int argc, char** argv) {
    int problem[3] = {128, 128, 128};
    for (int i = 1; i < argc && i < 4; ++i) {
        std::stringstream ss(argv[i]);
        ss >> problem[i - 1];
    }

    float scalars[2] = {1.0f, 0.0f};
    for (int i = 4; i < argc && i < 6; ++i) {
        std::stringstream ss(argv[i]);
        ss >> scalars[i - 4];
    }

    // Fix warnings: explicitly discard return values
    int device;
    (void)hipGetDevice(&device);
    hipDeviceProp_t deviceProp;
    (void)hipGetDeviceProperties(&deviceProp, device);

    std::cout << "Detected GPU: " << deviceProp.name
              << " (Arch: " << deviceProp.gcnArchName << ")" << std::endl;

    std::string arch = deviceProp.gcnArchName;
    auto starts_with = [](const std::string& str, const std::string& prefix) {
        return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
    };
    bool isSupported =
        starts_with(arch, "gfx906") ||
        starts_with(arch, "gfx908") ||
        starts_with(arch, "gfx90a") ||
        starts_with(arch, "gfx916") ||
        starts_with(arch, "gfx926") ||
        starts_with(arch, "gfx936");

    if (!isSupported) {
        std::cerr << "Warning: This GPU (" << arch
                  << ") may not support efficient FP16 GEMM." << std::endl;
        std::cerr << "For best results, use MI100/MI200/MI300 series." << std::endl;
    }

    //hipError_t result = TestHipBlasGemm<half>(
    hipError_t result = TestHipBlasGemm<float>(
        problem[0], problem[1], problem[2],
        scalars[0], scalars[1]
    );

    if (result == hipSuccess) {
        std::cout << "Passed." << std::endl;
    } else {
        std::cerr << "Failed with HIP error code: " << result << std::endl;
        return -1;
    }

    return 0;
}
