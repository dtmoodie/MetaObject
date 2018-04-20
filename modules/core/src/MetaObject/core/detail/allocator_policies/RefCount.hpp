#pragma once

namespace mo
{
    template<class Allocator>
    class RefCountPolicy: virtual public Allocator
    {
    public:
        ~RefCountPolicy();
        unsigned char* allocate(size_t num_bytes, size_t elem_size);

        void deallocate(unsigned char* ptr, size_t num_bytes);

        size_t refCount() const;
    private:
        size_t m_ref_count = 0;
    };
}