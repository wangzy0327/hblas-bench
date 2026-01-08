import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 配置参数
# ----------------------------
CSV_FILE = "csv/rocblas_gemm_performance_fp64_MI100.csv"
# CSV_FILE = "csv/rocblas_gemm_performance_fp64_Radeon(TM)ProVII.csv"
OUTPUT_PNG = "imgs/rocblas_gemm_fp64_performance_MI100.png"
# OUTPUT_PNG = "imgs/rocblas_gemm_fp64_performance_Radeon(TM)ProVII.png"
GPU_THEORETICAL_TFLOPS = 11.5  # 示例：MI100 FP64 峰值
# GPU_THEORETICAL_TFLOPS = 6.6  # 示例：MI50 FP64 峰值
GPU_NAME = "AMD Instinct MI100"
# GPU_NAME = "AMD Instinct MI50"

# 横轴范围
START = 256
END = 20480
STEP = 256
expected_sizes = set(range(START, END + 1, STEP))

# 关键尺寸标记（2 的幂或硬件对齐点）
key_sizes = [1024, 2048, 4096, 8192, 16384]

# ----------------------------
# 读取 CSV
# ----------------------------
df = pd.read_csv(CSV_FILE)

# 假设列名为：m,n,k,rocblas_time_ms,rocblas_gflops
# 我们只关心 m 和 rocblas_gflops
if 'm' not in df.columns or 'rocblas_gflops' not in df.columns:
    raise ValueError("CSV must contain columns: 'm' and 'rocblas_gflops'")

# 过滤出在 [256, 20480] 且是 256 倍数的行（可选）
df = df[df['m'].between(START, END)]
df = df[df['m'] % STEP == 0]

# 排序
df = df.sort_values('m')

sizes = df['m'].values
tflops = df['rocblas_gflops'].values / 1000.0  # 转为 TFLOPS

# ----------------------------
# 绘图
# ----------------------------
plt.figure(figsize=(14, 7))
plt.plot(sizes, tflops, 'bo-', markersize=5, linewidth=1.5, label='Measured FP64 GEMM Performance')

# 标注峰值点
peak_idx = np.argmax(tflops)
plt.annotate(
    f'Peak: {tflops[peak_idx]:.1f} TFLOPS\n@{sizes[peak_idx]}',
    xy=(sizes[peak_idx], tflops[peak_idx]),
    xytext=(20, 20),
    textcoords='offset points',
    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
    fontsize=10,
    color='red',
    weight='bold'
)

# 添加理论峰值线
plt.axhline(
    y=GPU_THEORETICAL_TFLOPS,
    color='r',
    linestyle='--',
    linewidth=1.5,
    label=f'Theoretical Peak ({GPU_NAME}: {GPU_THEORETICAL_TFLOPS} TFLOPS)'
)

# 标记关键尺寸（垂直虚线）
for size in key_sizes:
    if START <= size <= END:
        plt.axvline(x=size, color='gray', linestyle=':', alpha=0.7)
        plt.text(size, plt.ylim()[1] * 0.95, f'{size}', rotation=0,
                 verticalalignment='top', horizontalalignment='center',
                 fontsize=9, color='gray')

# 图表美化
plt.title('ROCm rocBLAS FP64 GEMM Performance (M = N = K)', fontsize=16, pad=20)
plt.xlabel('Matrix Dimension (M = N = K)', fontsize=14)
plt.ylabel('Performance (TFLOPS)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(START - STEP, END + STEP)
plt.xticks(np.arange(START, END + STEP, 2048), rotation=45)  # 每 2048 标一个 tick
plt.tight_layout()

# 图例
plt.legend(fontsize=12)

# 保存与显示
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"✅ 图表已保存为: {OUTPUT_PNG}")
plt.show()
