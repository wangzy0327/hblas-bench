#!/bin/bash

# 定义参数范围：256的整数倍，从256到6144
start=256
end=6144
step=256

# 固定alpha和beta
alpha=1.0
beta=0.0

# 测试程序路径（请根据实际路径修改）
EXECUTABLE="./test/gemm_blas_new_float"  # ← 请确保这是你编译出的可执行文件路径

# 输出结果文件
result_file="hipblas_gemm_performance_float.csv"

# 检查可执行文件是否存在
if [ ! -f "$EXECUTABLE" ]; then
    echo "错误: 可执行文件 $EXECUTABLE 不存在！"
    echo "请先编译 gemm_blas2.cpp 并生成该文件。"
    exit 1
fi

# 写入CSV表头
echo "m,n,k,hipblas_time_ms,hipblas_gflops" > "$result_file"

# 遍历 m = n = k 的方阵情况
for size in $(seq $start $step $end); do
    m=$size
    n=$size
    k=$size

    echo "========================================"
    echo "Testing shape: m=$m, n=$n, k=$k (GEMM via hipBLAS)"
    echo "========================================"
    
    # 执行测试程序，捕获全部输出（包括stderr）
    output=$("$EXECUTABLE" $m $n $k $alpha $beta 2>&1)
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "⚠️  程序执行失败（退出码: $exit_code），跳过此组数据。"
        echo "$output"
        # 写入空值行以保持 CSV 结构
        echo "$m,$n,$k,,"
        continue
    fi

    # 提取 hipBLAS 时间和 GFLOPS
    hipblas_time=$(echo "$output" | grep "Hipblas GEMM time:" | awk '{print $4}')
    hipblas_gflops=$(echo "$output" | grep "Hipblas GEMM Performance:" | awk '{print $4}')

    # 输出当前结果
    echo "hipBLAS: ${hipblas_time} ms | ${hipblas_gflops} GFLOPS"
    echo ""

    # 写入CSV（若提取失败则留空）
    echo "$m,$n,$k,$hipblas_time,$hipblas_gflops" >> "$result_file"
done

echo "========================================"
echo "✅ 测试完成！结果已保存至 $result_file"