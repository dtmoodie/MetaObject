#pragma once
#include "HelperMacros.hpp"
#include "MemoryBlock.hpp"
#include "MetaObject/detail/Export.hpp"
#include <boost/thread/mutex.hpp>
#include <cuda.h>
#include <list>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>

namespace mo
{
    MO_EXPORTS inline const unsigned char* alignMemory(const unsigned char* ptr, int elemSize);
    MO_EXPORTS inline unsigned char* alignMemory(unsigned char* ptr, int elemSize);
    MO_EXPORTS inline size_t alignmentOffset(const unsigned char* ptr, size_t elemSize);
    MO_EXPORTS void setScopeName(const std::string& name);
    MO_EXPORTS const std::string& getScopeName();

    class Allocator;

    class MO_EXPORTS Allocator : virtual public cv::cuda::GpuMat::Allocator, virtual public cv::MatAllocator
    {
      public:
        typedef std::shared_ptr<Allocator> Ptr;
        typedef std::shared_ptr<const Allocator> ConstPtr;
        static std::shared_ptr<Allocator> createAllocator();
        static void setDefaultAllocator(const std::shared_ptr<Allocator>& allocator);
        static std::shared_ptr<Allocator> getDefaultAllocator();

        // Used for stl allocators
        virtual unsigned char* allocateGpu(size_t num_bytes) = 0;
        virtual void deallocateGpu(uchar* ptr, size_t numBytes) = 0;

        virtual unsigned char* allocateCpu(size_t num_bytes) = 0;
        virtual void deallocateCpu(uchar* ptr, size_t numBytes) = 0;

        virtual void release() {}

        void setName(const std::string& name) { this->m_name = name; }
        const std::string& getName() { return m_name; }
      private:
        std::string m_name;
        static std::weak_ptr<Allocator> default_allocator;
    };

} // namespace mo
