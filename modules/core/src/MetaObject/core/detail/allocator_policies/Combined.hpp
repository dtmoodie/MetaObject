#pragma once

namespace mo
{
    template<class SmallAllocator, class LargeAllocator>
    class CombinedPolicy: virtual public SmallAllocator, virtual public LargeAllocator
    {
    public:
        CombinedPolicy(size_t threshold = 1 * 1024 * 512);

        unsigned char* allocate(size_t num_bytes, size_t elem_size);

        void deallocate(unsigned char* ptr, size_t num_bytes);
        void release();
        void setThreshold(size_t thresh);
        size_t getThreshold() const;
    private:
        size_t m_threshold;
    };

}