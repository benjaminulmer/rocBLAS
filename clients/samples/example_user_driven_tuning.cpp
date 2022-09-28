/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocblas.h"
#include "utility.hpp"
#include <hip/hip_runtime.h>

#include <chrono>
#include <map>

#define DIM1 64
#define DIM2 64
#define DIM3 10024

int main()
{
    rocblas_int       cold_calls = 1;
    rocblas_int       hot_calls  = 1;
    rocblas_operation transa = rocblas_operation_none, transb = rocblas_operation_transpose;
    float             alpha = 1.1, beta = 0.9;

    rocblas_int m = DIM1, n = DIM2, k = DIM3;
    rocblas_int lda, ldb, ldc, size_a, size_b, size_c;
    int         a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    rocblas_cout << "user driven tuning example" << std::endl;
    if(transa == rocblas_operation_none)
    {
        lda        = m;
        size_a     = k * lda;
        a_stride_1 = 1;
        a_stride_2 = lda;
    }
    else
    {
        lda        = k;
        size_a     = m * lda;
        a_stride_1 = lda;
        a_stride_2 = 1;
    }
    if(transb == rocblas_operation_none)
    {
        ldb        = k;
        size_b     = n * ldb;
        b_stride_1 = 1;
        b_stride_2 = ldb;
    }
    else
    {
        ldb        = n;
        size_b     = k * ldb;
        b_stride_1 = ldb;
        b_stride_2 = 1;
    }
    ldc    = m;
    size_c = n * ldc;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<float> ha(size_a + 1);
    std::vector<float> hb(size_b);
    std::vector<float> hc(size_c);
    std::vector<float> hc_gold(size_c);

    // initial data on host
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = rand() % 17;
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = rand() % 17;
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = rand() % 17;
    }
    hc_gold = hc;

    // allocate memory on device
    float *da, *db, *dc;
    CHECK_HIP_ERROR(hipMalloc(&da, size_a * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(float)));

    // copy matrices from host to device
    CHECK_HIP_ERROR(hipMemcpy(da, ha.data(), sizeof(float) * size_a, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(float) * size_b, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(float) * size_c, hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    rocblas_datatype type = rocblas_datatype_f32_r;

    // macros
#define GEMM_EX_ARGS                                                                             \
    handle, transa, transb, m, n, k, &alpha, da, type, lda, db, type, ldb, &beta, dc, type, ldc, \
        dc, type, ldc, type, rocblas_gemm_algo_standard
#define rocblas_gemm_exM(...) rocblas_gemm_ex(__VA_ARGS__)

    // Get number of solutions
    rocblas_int size;
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none, NULL, &size));
    rocblas_cout << size << " solution(s) found" << std::endl;

    // Fill array with list of solutions
    rocblas_int* ary = new rocblas_int[size];
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_ex_get_solutions(GEMM_EX_ARGS, rocblas_gemm_flags_none, ary, &size));

    // Example basic benchmark loop
    double      bestTime = std::numeric_limits<double>::max();
    rocblas_int bestSol  = -1;
    for(rocblas_int i = 0; i < size; ++i)
    {
        rocblas_int sol = ary[i];
        // warmup
        for(rocblas_int cc = 0; cc < cold_calls; ++cc)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS, sol, rocblas_gemm_flags_none));
        }
        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
        double time = get_time_us_sync(stream); // in microseconds

        // timing loop
        for(rocblas_int hc = 0; hc < hot_calls; ++hc)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS, sol, rocblas_gemm_flags_none));
        }
        time = get_time_us_sync(stream) - time;
        rocblas_cout << "Sol " << sol << ": " << time << " us" << std::endl;

        // track winner
        if(time < bestTime)
        {
            bestSol  = sol;
            bestTime = time;
        }
    }
    rocblas_cout << "Winner: " << bestSol << " in " << bestTime << " us" << std::endl;

    // Check if solution is valid for problem (fail case)
    rocblas_status check_fail = rocblas_gemm_exM(GEMM_EX_ARGS, 12, rocblas_gemm_flags_none);
    assert(check_fail == rocblas_status_invalid_solution_index);

    // Check if solution is valid for problem (success case)
    CHECK_ROCBLAS_ERROR(
        rocblas_gemm_exM(GEMM_EX_ARGS, bestSol, rocblas_gemm_flags_check_solution_index));

    // Solve using winner
    CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS, bestSol, rocblas_gemm_flags_none));

    // Solve using default solution
    CHECK_ROCBLAS_ERROR(rocblas_gemm_exM(GEMM_EX_ARGS, 0, rocblas_gemm_flags_none));

    return EXIT_SUCCESS;
}
