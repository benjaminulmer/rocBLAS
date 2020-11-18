/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_rotm.hpp"
#include "utility.hpp"

namespace
{
    constexpr int NB = 512;

    template <typename>
    constexpr char rocblas_rotm_name[] = "unknown";
    template <>
    constexpr char rocblas_rotm_name<float>[] = "rocblas_srotm_batched";
    template <>
    constexpr char rocblas_rotm_name<double>[] = "rocblas_drotm_batched";

    template <class T>
    rocblas_status rocblas_rotm_batched_impl(rocblas_handle handle,
                                             rocblas_int    n,
                                             T* const       x[],
                                             rocblas_int    incx,
                                             T* const       y[],
                                             rocblas_int    incy,
                                             const T* const param[],
                                             rocblas_int    batch_count)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

        auto layer_mode     = handle->layer_mode;
        auto check_numerics = handle->check_numerics;
        if(layer_mode & rocblas_layer_mode_log_trace)
            log_trace(handle, rocblas_rotm_name<T>, n, x, incx, y, incy, param, batch_count);
        if(layer_mode & rocblas_layer_mode_log_bench)
            log_bench(handle,
                      "./rocblas-bench -f rotm_batched -r",
                      rocblas_precision_string<T>,
                      "-n",
                      n,
                      "--incx",
                      incx,
                      "--incy",
                      incy,
                      "--batch_count",
                      batch_count);
        if(layer_mode & rocblas_layer_mode_log_profile)
            log_profile(handle,
                        rocblas_rotm_name<T>,
                        "N",
                        n,
                        "incx",
                        incx,
                        "incy",
                        incy,
                        "batch_count",
                        batch_count);

        if(n <= 0 || batch_count <= 0)
            return rocblas_status_success;

        if(!param)
            return rocblas_status_invalid_pointer;

        if(quick_return_param(handle, param, 0))
            return rocblas_status_success;

        if(!x || !y)
            return rocblas_status_invalid_pointer;
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status check_numerics_status
                = rocblas_check_numerics_vector_template(rocblas_rotm_name<T>,
                                                         handle,
                                                         n,
                                                         x,
                                                         0,
                                                         incx,
                                                         0,
                                                         batch_count,
                                                         check_numerics,
                                                         is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;

            check_numerics_status = rocblas_check_numerics_vector_template(rocblas_rotm_name<T>,
                                                                           handle,
                                                                           n,
                                                                           y,
                                                                           0,
                                                                           incy,
                                                                           0,
                                                                           batch_count,
                                                                           check_numerics,
                                                                           is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }

        rocblas_status status = rocblas_rotm_template<NB, true>(
            handle, n, x, 0, incx, 0, y, 0, incy, 0, param, 0, 0, batch_count);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status check_numerics_status
                = rocblas_check_numerics_vector_template(rocblas_rotm_name<T>,
                                                         handle,
                                                         n,
                                                         x,
                                                         0,
                                                         incx,
                                                         0,
                                                         batch_count,
                                                         check_numerics,
                                                         is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;

            check_numerics_status = rocblas_check_numerics_vector_template(rocblas_rotm_name<T>,
                                                                           handle,
                                                                           n,
                                                                           y,
                                                                           0,
                                                                           incy,
                                                                           0,
                                                                           batch_count,
                                                                           check_numerics,
                                                                           is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }
        return status;
    }

} // namespace

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCBLAS_EXPORT rocblas_status rocblas_srotm_batched(rocblas_handle     handle,
                                                    rocblas_int        n,
                                                    float* const       x[],
                                                    rocblas_int        incx,
                                                    float* const       y[],
                                                    rocblas_int        incy,
                                                    const float* const param[],
                                                    rocblas_int        batch_count)
try
{
    return rocblas_rotm_batched_impl(handle, n, x, incx, y, incy, param, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

ROCBLAS_EXPORT rocblas_status rocblas_drotm_batched(rocblas_handle      handle,
                                                    rocblas_int         n,
                                                    double* const       x[],
                                                    rocblas_int         incx,
                                                    double* const       y[],
                                                    rocblas_int         incy,
                                                    const double* const param[],
                                                    rocblas_int         batch_count)
try
{
    return rocblas_rotm_batched_impl(handle, n, x, incx, y, incy, param, batch_count);
}
catch(...)
{
    return exception_to_rocblas_status();
}

} // extern "C"
