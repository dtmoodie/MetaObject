#include "CudaContext.hpp"
#include "MetaObject/logging/logging.hpp"
#include <MetaObject/logging/profiling.hpp>
#include <cuda_runtime_api.h>
#ifdef HAVE_OPENCV
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif
namespace mo
{
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

    CudaContext::CudaContext(int priority) : m_priority(priority)
    {
        CUDA_ERROR_CHECK(cudaStreamCreateWithPriority(&m_cuda_stream, cudaStreamNonBlocking, priority))
            << " error creating stream";
    }
    void CudaContext::setName(const std::string& name)
    {
        mo::setStreamName(name.c_str(), getCudaStream());
        Context::setName(name);
    }
    cv::cuda::Stream& CudaContext::getStream()
    {
        THROW(warning) << "Not a cv::cuda::Stream context";
        return *static_cast<cv::cuda::Stream*>(nullptr);
    }

    cudaStream_t CudaContext::getCudaStream() const { return m_cuda_stream; }

    void CudaContext::setStream(const cv::cuda::Stream& stream)
    {
        if (this->m_cuda_stream)
        {
            CUDA_ERROR_CHECK(cudaStreamDestroy(this->m_cuda_stream)) << "Unable to cleanup stream";
        }
        this->m_cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
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
