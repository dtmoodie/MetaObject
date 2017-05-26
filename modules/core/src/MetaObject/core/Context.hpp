#pragma once
#include "MetaObject/detail/Export.hpp"
#include <string>
#include <memory>
typedef struct CUstream_st *cudaStream_t;
namespace cv{
namespace cuda{
    class Stream;
} // namespace cv::cuda
} // namespace cv
namespace mo{
    class Allocator;
    class MO_EXPORTS Context{
    public:
        static std::shared_ptr<Context> getDefaultThreadContext();
        static void                     setDefaultThreadContext(const std::shared_ptr<Context>& ctx);
        static std::shared_ptr<Context> create(const std::string& name = "");

        virtual                         ~Context();
        virtual void                    setName(const std::string& name);
        virtual cv::cuda::Stream&       getStream();
        virtual cudaStream_t            getCudaStream() const;
        virtual void                    setStream(const cv::cuda::Stream& stream);

        size_t process_id = 0;
        size_t thread_id = 0;
        std::string host_name;
        Allocator* allocator;
    protected:
        Context();
        std::string name;
    }; // class mo::Context
} // namespace mo
