#include "CudaContext.hpp"
#include "MetaObject/logging/logging.hpp"
#include <MetaObject/logging/profiling.hpp>
#include <cuda_runtime_api.h>
#ifdef HAVE_OPENCV
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif
namespace mo{
    CudaContext::CudaContext(){
        CUDA_ERROR_CHECK(cudaStreamCreate(&stream)) << " error creating stream";
    }
    void CudaContext::setName(const std::string& name){
        mo::setStreamName(name.c_str(), getCudaStream());
        Context::setName(name);
    }
    cv::cuda::Stream& CudaContext::getStream(){
        THROW(warning) << "Not a gpu context";
        return *static_cast<cv::cuda::Stream*>(nullptr);
    }
    cudaStream_t CudaContext::getCudaStream() const{
        return stream;
    }
    void CudaContext::setStream(const cv::cuda::Stream& stream){
        if(this->stream){
            CUDA_ERROR_CHECK(cudaStreamDestroy(this->stream)) << "Unable to cleanup stream";
        }
        this->stream = cv::cuda::StreamAccessor::getStream(stream);
    }
}


