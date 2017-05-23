#include "Context.hpp"


namespace mo{
    class MO_EXPORTS CudaContext: public Context{
    public:
        CudaContext();
        virtual void                    setName(const std::string& name);
        virtual cv::cuda::Stream&       getStream();
        virtual cudaStream_t            getCudaStream() const;
        virtual void                    setStream(const cv::cuda::Stream& stream);
    private:
        cudaStream_t stream;
    };
}
