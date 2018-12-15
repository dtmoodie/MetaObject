#ifndef MO_CUDA_ALLOCATOR_HPP
#define MO_CUDA_ALLOCATOR_HPP
#include <cstdint>
namespace mo
{
    namespace cuda
    {
        struct TextureAligned
        {
            TextureAligned(int32_t device_id = -1);
            void sizeNeeded(int rows, int cols, int elemSize, uint64_t& size_needed, uint64_t& stride);

          private:
            uint32_t m_texture_alignment;
        };
    }
}

#endif // MO_CUDA_ALLOCATOR_HPP
