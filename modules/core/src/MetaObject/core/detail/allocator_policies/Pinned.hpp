#pragma once
#include <cstdint>

namespace mo
{
    template <class XPU>
    class PinnedPolicy
    {
      public:
        unsigned char* allocate(std::size_t num_bytes, std::size_t elem_size);

        void deallocate(unsigned char* ptr, std::size_t num_bytes);

        void release();
    };
}
