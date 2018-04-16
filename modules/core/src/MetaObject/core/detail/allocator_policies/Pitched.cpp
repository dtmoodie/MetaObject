#include "Pitched.hpp"
#include <MetaObject/core/metaobject_config.hpp>
#if MO_OPENCV_HAVE_CUDA
#include <opencv2/core/cuda.hpp>
#endif
namespace mo
{
    PitchedPolicy::PitchedPolicy()
    {
#if MO_OPENCV_HAVE_CUDA
        texture_alignment = cv::cuda::DeviceInfo(cv::cuda::getDevice()).textureAlignment();
#else
        texture_alignment = 1;
#endif
    }

    void PitchedPolicy::sizeNeeded(int rows, int cols, int elemSize, size_t& size_needed, size_t& stride)
    {
        if (rows == 1 || cols == 1)
        {
            stride = static_cast<size_t>(cols * elemSize);
        }
        else
        {
            if ((static_cast<size_t>(cols * elemSize) % texture_alignment) == 0)
            {
                stride = static_cast<size_t>(cols * elemSize);
            }
            else
            {
                stride = static_cast<size_t>(cols * elemSize) + texture_alignment -
                    (static_cast<size_t>(cols * elemSize) % texture_alignment);
            }
        }
        size_needed = stride * static_cast<size_t>(rows);
    }
}