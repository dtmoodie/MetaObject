#pragma once

namespace mo
{
    template<class Allocator>
    struct UsagePolicy: public Allocator
    {
        unsigned char* allocate(size_t num_bytes, size_t elem_size);

        void deallocate(unsigned char* ptr, size_t num_bytes);

        size_t usage() const;

    protected:
        size_t m_usage = 0;
    };
}