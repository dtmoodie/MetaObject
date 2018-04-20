#pragma once
#include <MetaObject/core/metaobject_config.hpp>
#include "MetaObject/core.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <memory>
#include <string>
typedef struct CUstream_st* cudaStream_t;
namespace cv
{
    namespace cuda
    {
        class Stream;
    } // namespace cv::cuda
} // namespace cv
namespace mo
{
    class Context;
    class CvContext;

#if MO_OPENCV_HAVE_CUDA
    using ContexTypes = std::tuple<Context, CvContext>;
#else
    using ContexTypes = std::tuple<Context>;
#endif
    class Allocator;
    class MO_EXPORTS Context
    {
      public:
        /*!
         * \brief creates a Context based on the underlying hardware
         * \param name of the context
         * \param priority of the created cuda stream
         * \return shared ptr to created context
         */
        static std::shared_ptr<Context> create(const std::string& name = "", int priority = 5);
        static Context* getCurrent();
        static void setCurrent(Context* ctx);

        Context();
        virtual ~Context();
        virtual cv::cuda::Stream& getStream();
        virtual cudaStream_t getCudaStream() const;

        virtual void setStream(const cv::cuda::Stream& stream);
        virtual void setStream(cudaStream_t stream);

        virtual void setName(const std::string& name);
        std::string getName() const { return name; }
        size_t getThreadId() const { return thread_id; }
        inline bool isDeviceContext() { return device_id != -1; }

        size_t process_id = 0;
        size_t thread_id = 0;
        int device_id = -1;
        std::string host_name;
        std::shared_ptr<Allocator> allocator;
        // Type of derived class, used for type switch
        mo::TypeInfo context_type;

      protected:
        std::string name;
    }; // class mo::Context
} // namespace mo
