#pragma once
#include "MetaObject/core.hpp"
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

        virtual ~Context();
        virtual cv::cuda::Stream& getStream();
        virtual cudaStream_t getCudaStream() const;

        virtual void setStream(const cv::cuda::Stream& stream);
        virtual void setStream(cudaStream_t stream);

        virtual void setName(const std::string& name);
        std::string getName() const { return name; }
        size_t getThreadId() const { return thread_id; }

        size_t process_id = 0;
        size_t thread_id = 0;
        std::string host_name;
        std::shared_ptr<Allocator> allocator;

      protected:
        Context();
        std::string name;
        static thread_local Context* current_context;
    }; // class mo::Context
} // namespace mo
