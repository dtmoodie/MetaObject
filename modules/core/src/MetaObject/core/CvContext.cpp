#include "CvContext.hpp"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <MetaObject/logging/profiling.hpp>
namespace mo{
    CvContext::CvContext(){

    }

    void CvContext::setName(const std::string& name){
        mo::setStreamName(name.c_str(), getCudaStream());
        Context::setName(name);
    }

    cv::cuda::Stream& CvContext::getStream(){
        return stream;
    }

    cudaStream_t CvContext::getCudaStream() const{
        return cv::cuda::StreamAccessor::getStream(stream);
    }

    void CvContext::setStream(const cv::cuda::Stream& stream){
        this->stream = stream;
        mo::setStreamName(name.c_str(), getCudaStream());
    }
}
