/*
#include "CudaContext.hpp"
#if MO_HAVE_CUDA
#include "MetaObject/logging/logging.hpp"
#include <MetaObject/logging/profiling.hpp>
#include <cuda_runtime_api.h>
#ifdef MO_HAVE_OPENCV
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

namespace mo
{
    CudaContext::CudaContext(TypeInfo type, int priority)
        : Context(type)
        , m_priority(priority)
    {
        init();
    }

    CudaContext::CudaContext(int priority)
        : m_priority(priority)
    {
        init();
    }

    CudaContext::~CudaContext()
    {
        {
            CUDA_ERROR_CHECK(cudaStreamSynchronize(this->m_cuda_stream));
        }
        {
            CUDA_ERROR_CHECK(cudaStreamDestroy(this->m_cuda_stream));
        }
        this->m_cuda_stream = nullptr;
    }

    void CudaContext::init()
    {
        {
            CUDA_ERROR_CHECK(cudaStreamCreateWithPriority(&m_cuda_stream, cudaStreamNonBlocking, m_priority))
                << " error creating stream";
        }
        {
            int device_id;
            CUDA_ERROR_CHECK(cudaGetDevice(&device_id));
            setDeviceId(device_id);
        }
    }

    void CudaContext::setName(const std::string& name)
    {
        setStreamName(name.c_str(), getCudaStream());
        Context::setName(name);
    }

    cv::cuda::Stream& CudaContext::getStream()
    {
#ifdef MO_HAVE_OPENCV
        static thread_local cv::cuda::Stream stream;
        stream = cv::cuda::StreamAccessor::wrapStream(m_cuda_stream);
        return stream;
#else
        THROW(warning) << "Not a cv::cuda::Stream context";
        return *static_cast<cv::cuda::Stream*>(nullptr);
#endif
    }

    cudaStream_t CudaContext::getCudaStream() const
    {
        return m_cuda_stream;
    }

    void CudaContext::setStream(const cv::cuda::Stream& stream)
    {
        if (this->m_cuda_stream)
        {
            CUDA_ERROR_CHECK(cudaStreamDestroy(this->m_cuda_stream)) << "Unable to cleanup stream";
        }
#if MO_OPENCV_HAVE_CUDA
        this->m_cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
#else
        THROW(warning) << "Not built against opencv that's compatible with CUDA";
#endif
    }

    void CudaContext::setStream(cudaStream_t stream)
    {
        if (this->m_cuda_stream)
        {
            CUDA_ERROR_CHECK(cudaStreamDestroy(this->m_cuda_stream)) << "Unable to cleanup stream";
        }
        this->m_cuda_stream = stream;
    }
}
#endif
*/
