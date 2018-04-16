#include "../MemoryBlock.hpp"
#include <list>
#include <memory>
namespace mo
{
    template<class XPU>
    class PoolPolicy
    {
    public:
        PoolPolicy();
        PoolPolicy(const PoolPolicy&) = delete;
        PoolPolicy& operator = (const PoolPolicy&) = delete;

        unsigned char* allocate(size_t num_bytes, size_t elem_size);

        void deallocate(unsigned char* ptr, size_t num_bytes);

        void release();
    private:
        size_t m_initial_block_size;
        std::list < std::unique_ptr<MemoryBlock<XPU> > > m_blocks;
    };
}