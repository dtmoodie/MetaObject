#pragma once
#include "MetaObject/detail/Export.hpp"
#include <memory>
#include <string>
namespace mo
{
    MO_EXPORTS const unsigned char* alignMemory(const unsigned char* ptr, int elemSize);
    MO_EXPORTS unsigned char* alignMemory(unsigned char* ptr, int elemSize);
    MO_EXPORTS size_t alignmentOffset(const unsigned char* ptr, size_t elemSize);
    MO_EXPORTS void setScopeName(const std::string& name);
    MO_EXPORTS const std::string& getScopeName();

    class Allocator;

    class MO_EXPORTS Allocator
    {
      public:
        virtual ~Allocator();
        typedef std::shared_ptr<Allocator> Ptr;
        typedef std::shared_ptr<const Allocator> ConstPtr;

        static std::shared_ptr<Allocator> createDefaultOpencvAllocator();
        static std::shared_ptr<Allocator> createAllocator();
        static void setDefaultAllocator(const std::shared_ptr<Allocator>& allocator);
        static std::shared_ptr<Allocator> getDefaultAllocator();

        // Used for stl allocators
        virtual unsigned char* allocateGpu(size_t num_bytes, size_t element_size = 1) = 0;
        virtual void deallocateGpu(unsigned char* ptr, size_t num_bytes) = 0;

        virtual unsigned char* allocateCpu(size_t num_bytes, size_t element_size = 1) = 0;
        virtual void deallocateCpu(unsigned char* ptr, size_t num_bytes) = 0;

        virtual void release();

        void setName(const std::string& name);
        const std::string& name() const;

      private:
        std::string m_name;
        static std::weak_ptr<Allocator> default_allocator;
    };

} // namespace mo
