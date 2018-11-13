#pragma once
#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include <opencv2/core/cuda.hpp>
#if MO_HAVE_CUDA
#include "Context.hpp"
#include "CudaContext.hpp"
namespace mo
{
    class CvContext : public CudaContext
    {
      public:
        CvContext(int priority);
        virtual ~CvContext() override;
        virtual void setName(const std::string& name);
        virtual cv::cuda::Stream& getStream();
        virtual cudaStream_t getCudaStream() const;
        virtual void setStream(const cv::cuda::Stream& stream);
        virtual void setStream(cudaStream_t stream);

      protected:
        CvContext(TypeInfo type);
        cv::cuda::Stream m_cv_stream;
    };
}
#endif
#endif
