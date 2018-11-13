#pragma once

#include "Context.hpp"

namespace cv
{
    namespace cuda
    {
        class Stream;
    }
}

namespace mo
{
    namespace cuda
    {
        struct CvContext: virtual public cuda::Context
        {
            virtual cv::cuda::Stream& cvStream();
        };
    }
}
