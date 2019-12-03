#include "Continuous.hpp"

namespace mo
{
    void ContinuousPolicy::sizeNeeded(int rows, int cols, int elemSize, size_t& size_needed, size_t& stride)
    {
        stride = static_cast<size_t>(cols) * static_cast<size_t>(elemSize);
        size_needed = stride * static_cast<size_t>(rows);
    }
}
