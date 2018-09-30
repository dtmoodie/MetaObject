#pragma once
#include <boost/thread/mutex.hpp>
namespace mo
{
    template<class Allocator>
    class LockPolicy:public Allocator
    {
    public:
        unsigned char* allocate(size_t num_bytes, size_t elem_size);

        void deallocate(unsigned char* ptr, size_t num_bytes);
    private:
        boost::mutex m_mtx;
    };
}