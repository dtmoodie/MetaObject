#pragma once
#include "Context.hpp"

namespace mo {
class MO_EXPORTS CudaContext : public Context {
public:
    ~CudaContext();
    CudaContext(int priority);
    virtual void setName(const std::string& name);
    virtual cv::cuda::Stream& getStream();
    virtual cudaStream_t      getCudaStream() const;
    virtual void setStream(const cv::cuda::Stream& stream);
    virtual void setStream(cudaStream_t stream);

protected:
    cudaStream_t m_cuda_stream;
    int          m_priority;
};
}
