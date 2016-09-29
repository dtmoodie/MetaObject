#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <string>
namespace cv
{
    namespace cuda
    {
        class Stream;
    }
}

namespace mo
{
    class MO_EXPORTS Context
    {
    public:
        Context();
        size_t process_id = 0;
        size_t thread_id = 0;
        std::string host_name;
        cv::cuda::Stream* stream = nullptr;
    };
}