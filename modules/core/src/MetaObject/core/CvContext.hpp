#include "Context.hpp"
#include <opencv2/core/cuda.hpp>
namespace mo{
    class CvContext: public Context{
    public:
        CvContext();
        virtual void                    setName(const std::string& name);
        virtual cv::cuda::Stream&       getStream();
        virtual cudaStream_t            getCudaStream() const;
        virtual void                    setStream(const cv::cuda::Stream& stream);
    private:
        cv::cuda::Stream stream;
    };
}
