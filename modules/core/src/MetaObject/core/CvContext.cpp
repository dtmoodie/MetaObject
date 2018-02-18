#include "CvContext.hpp"
#include <MetaObject/logging/profiling.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <MetaObject/logging/logging.hpp>
namespace mo
{
    CvContext::CvContext(int priority)
        : CudaContext(priority), m_cv_stream(cv::cuda::StreamAccessor::wrapStream(this->m_cuda_stream))
    {
    }

    CvContext::~CvContext() 
    { 
        try
        {
            m_cv_stream.waitForCompletion();
        }
        catch (cv::Exception& /*e*/)
        {
            MO_LOG(error) << "Trying to delete a context after cuda context destruction.  This could be caused by forgetting to cleanup any thread_local contexts.  Fix this by calling mo::Context::setDefaultThreadContext({}); before program exit to cleanup any dangling contexts before driver shutdown.";
        }
    }

    void CvContext::setName(const std::string& name)
    {
        mo::setStreamName(name.c_str(), getCudaStream());
        Context::setName(name);
    }

    cv::cuda::Stream& CvContext::getStream() { return m_cv_stream; }

    cudaStream_t CvContext::getCudaStream() const { return this->m_cuda_stream; }

    void CvContext::setStream(const cv::cuda::Stream& stream)
    {
        CudaContext::setStream(cv::cuda::StreamAccessor::getStream(stream));
        this->m_cv_stream = stream;
        mo::setStreamName(name.c_str(), getCudaStream());
    }

    void CvContext::setStream(cudaStream_t stream)
    {
        CudaContext::setStream(stream);
        this->m_cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
        mo::setStreamName(name.c_str(), stream);
    }
}
