#pragma once
#include <MetaObject/detail/Export.hpp>
#include <ctime>
#include <list>
#include <utility>

namespace mo
{

    class MO_EXPORTS CpuMemoryStack
    {
      public:
        CpuMemoryStack(size_t delay);
        virtual ~CpuMemoryStack();
        virtual bool allocate(void** ptr, size_t total, size_t elemSize);
        unsigned char* allocate(size_t total);
        virtual bool deallocate(void* ptr, size_t total);

      private:
        void cleanup(bool force, bool dtor);
        size_t total_usage = 0;
        size_t deallocation_delay;
        std::list<std::tuple<unsigned char*, clock_t, size_t>> deallocate_stack;
    };
}
