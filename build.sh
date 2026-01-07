#hipcc --offload-arch=gfx908  -o ./test/gemm_blas ./src/gemm_blas.cpp -lhipblas
hipcc -O3 -ffast-math --offload-arch=gfx908  -o ./test/gemm_blas2 ./src/gemm_blas2.cpp -lhipblas
hipcc -std=c++17 -O3 -ffast-math --offload-arch=gfx908 -Wno-return-type src/gemm_blas3.cpp -o ./test/gemm_blas_new -lhipblas
hipcc -std=c++17  --offload-arch=gfx906 -Wno-return-type src/gemm_blas3.cpp -o ./test/gemm_blas_new_float -lhipblas
hipcc -std=c++17  --offload-arch=gfx908 -Wno-return-type src/gemm_blas3.cpp -o ./test/gemm_blas_new_float -lhipblas
hipcc -std=c++17  --offload-arch="gfx90a:sramecc-:xnack-" -Wno-return-type src/gemm_blas3.cpp -o ./test/gemm_blas_new_float -lhipblas
hipcc -std=c++11  --offload-arch=gfx908 -Wno-return-type -I/opt/rocm/include/rocblas src/gemm_blas3_new.cpp -o ./test/gemm_rocblas -lrocblas
hipcc -std=c++11  --offload-arch=gfx908 -Wno-return-type -I/opt/rocm/include/rocblas src/gemm_blas3_new.cpp -o ./test/gemm_rocblas_fp32 -lrocblas
hipcc -o ./test/gemm ./src/gemm.cpp
hipcc -o ./test/gemv ./src/gemv.cpp
echo "finish build, all execution file in test"
