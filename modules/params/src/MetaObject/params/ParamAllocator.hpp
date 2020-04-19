#ifndef MO_PARAMS_PARAMALLOCATOR_HPP
#define MO_PARAMS_PARAMALLOCATOR_HPP
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/detail/Export.hpp>
#include <MetaObject/thread/Mutex.hpp>

#include <list>
#include <memory>
namespace mo
{
    using ct::ptrCast;
    struct MO_EXPORTS ParamAllocator
    {
        using Ptr_t = std::shared_ptr<ParamAllocator>;
        using ConstPtr_t = std::shared_ptr<const ParamAllocator>;
        struct MO_EXPORTS SerializationBuffer : public ct::TArrayView<uint8_t>
        {
            SerializationBuffer(ParamAllocator& alloc, uint8_t* begin, size_t sz);

            SerializationBuffer(ParamAllocator& alloc, uint8_t* begin, uint8_t* end);

            SerializationBuffer(const SerializationBuffer&) = delete;
            SerializationBuffer(SerializationBuffer&&) noexcept = default;
            SerializationBuffer& operator=(const SerializationBuffer&) = delete;
            SerializationBuffer& operator=(SerializationBuffer&&) noexcept = default;

            ~SerializationBuffer();

          private:
            ParamAllocator& m_alloc;
        };

        static Ptr_t create(Allocator::Ptr_t allocator = Allocator::getDefault());

        ParamAllocator(Allocator::Ptr_t allocator = Allocator::getDefault());
        ~ParamAllocator();

        void setPadding(size_t header, size_t footer = 0);

        template <class T>
        T* allocate(size_t num)
        {
            Mutex::Lock_t lock(m_mtx);
            auto allocation = allocateImpl(num, sizeof(T));
            return ptrCast<T>(allocation.requested);
        }

        template <class T>
        std::shared_ptr<SerializationBuffer> allocateSerialization(size_t header_sz, size_t footer_sz, const T* ptr)
        {
            return allocateSerializationImpl(header_sz, footer_sz, static_cast<const void*>(ptr), sizeof(T));
        }

        template <class T>
        void deallocate(T* ptr)
        {
            deallocateImpl(static_cast<void*>(ptr));
        }

        Allocator::Ptr_t getAllocator() const;

        void setAllocator(Allocator::Ptr_t allocator);

      private:
        void deallocateImpl(void* ptr);

        std::shared_ptr<SerializationBuffer>
        allocateSerializationImpl(size_t header_sz, size_t footer_sz, const void* ptr, size_t elem_size);

        struct CurrentAllocations
        {
            uint8_t* begin = nullptr;
            uint8_t* requested = nullptr;
            uint8_t* end = nullptr;
            size_t requested_size = 0;
            int ref_count = 1;
        };

        CurrentAllocations allocateImpl(size_t num, size_t elem_size);

        // This is the allocator used for actual allocations
        // IE pinned memory or shared
        // TODO weak_ptr?
        Allocator::Ptr_t m_allocator;
        size_t m_header_pad = 0;
        size_t m_footer_pad = 0;
        mutable Mutex m_mtx;
        std::list<CurrentAllocations> m_allocations;
    };
} // namespace mo
#endif // MO_PARAMS_PARAMALLOCATOR_HPP