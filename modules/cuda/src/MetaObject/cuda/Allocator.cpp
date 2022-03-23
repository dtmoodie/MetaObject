#include "Allocator.hpp"
#include <cuda_runtime_api.h>

namespace mo
{
    namespace cuda
    {
        TextureAligned::TextureAligned(int32_t device_id)
        {
            if (device_id == -1)
            {
                cudaGetDevice(&device_id);
            }
            int32_t texture_alignment;
            cudaDeviceGetAttribute(&texture_alignment, cudaDevAttrTextureAlignment, device_id);
            m_texture_alignment = static_cast<uint32_t>(texture_alignment);
        }

        void TextureAligned::sizeNeeded(int rows, int cols, int elemSize, uint64_t& size_needed, uint64_t& stride)
        {
            if (rows == 1 || cols == 1)
            {
                stride = static_cast<uint64_t>(cols * elemSize);
            }
            else
            {
                if ((static_cast<uint64_t>(cols * elemSize) % m_texture_alignment) == 0)
                {
                    stride = static_cast<uint64_t>(cols * elemSize);
                }
                else
                {
                    stride = static_cast<uint64_t>(cols * elemSize) + m_texture_alignment -
                             (static_cast<uint64_t>(cols * elemSize) % m_texture_alignment);
                }
            }
            size_needed = stride * static_cast<uint64_t>(rows);
        }
    } // namespace cuda
} // namespace mo
