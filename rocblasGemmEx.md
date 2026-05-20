# 详解GemmEx调用

cublasGemmEx和rocblas_gemm_ex函数解析

列出cublasGemmEx函数的参数

```c++
cublasGemmEx(handle,transa,transb,m,n,k,alpha,A,Atype,lda,B,Btype,ldb,beta,C,Ctype,ldc,computeType,algo)
```

rocblas_gemm_ex函数的参数

```c++
rocblas_gemm_ex(handle,transA,transB,m,n,k,alpha,A,a_type,lda,B,b_type,ldb,beta,C,c_type,ldc,D,dtype,ldd,compute_type,algo,solution_index,flags)
```

举例 (m=2,n=4,k=3)  C = AB
$$
A =
\begin{pmatrix}
1&2&3\\
4&5&6\\
\end{pmatrix}
$$

$$
B = 
\begin{pmatrix}
1&2&3&4\\
5&6&7&8\\
9&10&11&12\\
\end{pmatrix}
$$

$$
C = AB = 
\begin{pmatrix}
38&44&50&56\\
83&98&113&128\\
\end{pmatrix}
$$

矩阵在内存里都是一维顺序的，以矩阵A举例（1,2,3,4,5,6）。默认在cublas和rocblas里都是列主序，当调用cublasGemmEx或者rocblas_gemm_ex时，A，B在传入函数后，其内存排布就变成下面这样。
$$
输入 A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \xrightarrow{\text{内存排布}} \{1\ 2\ 3\ 4\ 5\ 6\} \xrightarrow[\text{列主序}]{lda=m=2} A_1 = \begin{pmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{pmatrix}
$$

$$
B = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \end{pmatrix} \xrightarrow{\text{内存排布}} \{1\ 2\ 3\ 4\ 5\ 6\ ...\ 11\ 12\} \xrightarrow[\text{列主序}]{ldb=k=3} B_1 = \begin{pmatrix} 1 & 4 & 7 & 10 \\ 2 & 5 & 8 & 11 \\ 3 & 6 & 9 & 12 \end{pmatrix}
$$



按照上面这样的A，B排布，很明显是计算不出正确的C矩阵（补充：这样计算出的C矩阵也不是转置就正确的）。上面这样的排布调用的参数以cublasGemmEx 举例（m=2，n=4，k=3）。错误写法如下：

```cpp
cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,alpha,A,CUDA_R_16F,m,B,CUDA_R_16F,k,beta,C,CUDA_R_16F,m,computeType,CUDA_R_32F)
```

```cpp
rocblas_gemm_ex(handle,rocblas_operation_none,rocblas_operation_none,m,n,k,alpha,A,rocblas_datatype_f16_r,m,B,rocblas_datatype_f16_r,k,beta,C,rocblas_datatype_f16_r,m,D,rocblas_datatype_f16_r,m,rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0)
```

既然这样按照默认非转置的列排序是错误的，那么可以将A尝试使用转置来进行计算。这样的内存排布结构如下：
$$
输入 A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \xrightarrow{\text{内存排布}} \{1\ 2\ 3\ 4\ 5\ 6\} \xrightarrow[\text{列主序}]{lda=k=3} A_1 = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix} \xrightarrow{\text{转置}} A_2 = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}
$$

$$
输入 B = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \end{pmatrix} \xrightarrow{\text{内存排布}} \{1\ 2\ 3\ 4\ 5\ 6\ ...\ 11\ 12\} \xrightarrow[\text{列主序}]{ldb=n=4} B_1 = \begin{pmatrix} 1 & 5 & 9 \\ 2 & 6 & 10 \\ 3 & 7 & 11 \\ 4 & 8 & 12 \end{pmatrix} \xrightarrow{\text{转置}} B_2 = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \end{pmatrix}
$$



很明显，如果按照上面这样排布，那么计算得到的结果就是正确的C矩阵。因此计算得到的C如下：

$$
C_2 = A_2^T B_2^T = \begin{pmatrix} 38 & 44 & 50 & 56 \\ 83 & 98 & 113 & 128 \end{pmatrix} \xrightarrow{\text{内存排布}} \{38\ 83\ 44\ 98\ ...\ 56\ 128\} \xrightarrow[\text{行主序}]{ldc=m=2} \begin{pmatrix} 38 & 83 \\ 44 & 98 \\ 50 & 113 \\ 56 & 128 \end{pmatrix}
$$
但是有一点需要注意计算得到的C矩阵的结果是左边这样，但是由于C也是列主序的，所以内存里的排布不是按照（38,44,50,56,83,98,113,128）这样行主序排序的。而是列主序（38,83,44,98,50,113,56,128）这样排序的。所以C的计算结果指针需要进行转置转换为右侧那样，这样得到的C矩阵的排布才是我们需要的。因为A,B需要转置，所以A(m x k)的lda就为k，B(k x n)的ldb为n，C(m x n)的ldc 还是为m，但是C的结果计算结果需要转置才是正确的C。

上面这样转置计算出正确结果的函数调用，以cublasGemmEx 举例（m=2，n=4，k=3）如下：

```cpp
cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_T,m,n,k,alpha,A,CUDA_R_16F,k,B,CUDA_R_16F,n,beta,C,CUDA_R_16F,m,computeType,CUDA_R_32F)
transpose_gemm(C_in,C_out,m_width,n_height)
```

```cpp
rocblas_gemm_ex(handle,rocblas_operation_transpose,rocblas_operation_transpose,m,n,k,alpha,A,rocblas_datatype_f16_r,k,B,rocblas_datatype_f16_r,n,beta,C,rocblas_datatype_f16_r,m,D,rocblas_datatype_f16_r,m,rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0)
transpose_gemm(C_in,C_out,m_width,n_height)
```

上面采取转置的方式计算得到的结果还需要转置才是我们需要的，比较麻烦。我们可以根据矩阵相乘转置的性质进行直接得到结果。C^T=(AB)^T=(B^T A^T)

这样的内存排布结构如下：
$$
输入 A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} \xrightarrow{\text{内存排布}} \{1\ 2\ 3\ 4\ 5\ 6\} \xrightarrow[\text{列主序}]{lda=k=3} A_1 = \begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}
$$

$$
输入 B = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \end{pmatrix} \xrightarrow{\text{内存排布}} \{1\ 2\ 3\ 4\ 5\ 6\ \dots\ 11\ 12\} \xrightarrow[\text{列主序}]{ldb=n=4} B_1 = \begin{pmatrix} 1 & 5 & 9 \\ 2 & 6 & 10 \\ 3 & 7 & 11 \\ 4 & 8 & 12 \end{pmatrix}
$$

$$
C_1 = B_1 A_1 = \begin{pmatrix} 38 & 83 \\ 44 & 98 \\ 50 & 113 \\ 56 & 128 \end{pmatrix} \xrightarrow{\text{内存排布}} \{38\ 44\ 50\ 56\ \dots\ 113\ 128\} \xrightarrow[\text{行主序}]{\text{ldc=n=4}} \begin{pmatrix} 38 & 83 \\ 44 & 98 \\ 50 & 113 \\ 56 & 128 \end{pmatrix}
$$

计算后的结果即为最后正确的C，上面这样非转置计算出正确结果的函数调用，以cublasGemmEx 举例（m=2，n=4，k=3）如下：

```cpp
cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,alpha,B,CUDA_R_16F,n,A,CUDA_R_16F,k,beta,C,CUDA_R_16F,n,computeType,CUDA_R_32F)
```

```cpp
rocblas_gemm_ex(handle,rocblas_operation_none,rocblas_operation_none,n,m,k,alpha,B,rocblas_datatype_f16_r,n,A,rocblas_datatype_f16_r,k,beta,C,rocblas_datatype_f16_r,n,D,rocblas_datatype_f16_r,n,rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0)
```

