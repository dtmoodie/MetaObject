#pragma once
#include <MetaObject/core/metaobject_config.hpp>
#include "Allocator.hpp"
#include "MetaObject/logging/logging.hpp"
#include <MetaObject/thread/cuda.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/tss.hpp>
#if MO_HAVE_CUDA
#include <cuda_runtime.h>
#endif
#if MO_HAVE_OPENCV
#include <opencv2/core/cuda.hpp>
#endif
#include <unordered_map>

#define MO_CUDA_ERROR_CHECK(expr, msg)                                                                             \
{                                                                                                                  \
    cudaError_t err = (expr);                                                                                      \
    if (err != cudaSuccess)                                                                                        \
    {                                                                                                              \
        THROW(warning) << #expr << " failed " << cudaGetErrorString(err) << " " msg;                               \
    }                                                                                                              \
}

namespace mo
{

}
