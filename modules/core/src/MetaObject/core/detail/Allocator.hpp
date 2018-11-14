#pragma once
#include "MetaObject/detail/Export.hpp"
#include <memory>
#include <string>
namespace mo
{

    class Allocator;

    class MO_EXPORTS Allocator
    {
      public:
        virtual ~Allocator();
        typedef std::shared_ptr<Allocator> Ptr;
        typedef std::shared_ptr<const Allocator> ConstPtr;

        virtual void* allocate(const uint64_t num_bytes, const uint64_t element_size = 1) = 0;
        virtual void deallocate(void* ptr, const uint64_t num_bytes) = 0;

        virtual void release();

        void setName(const std::string& name);
        const std::string& name() const;

      private:
        std::string m_name;
    };

} // namespace mo
