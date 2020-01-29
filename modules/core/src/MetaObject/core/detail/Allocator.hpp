#pragma once
#include <MetaObject/detail/Export.hpp>

#include <ct/types/TArrayView.hpp>

#include <memory>
#include <string>

namespace mo
{
    struct MO_EXPORTS Allocator
    {
        using Ptr_t = std::shared_ptr<Allocator>;
        using ConstPtr_t = std::shared_ptr<const Allocator>;

        static std::shared_ptr<Allocator> getDefault();
        static void setDefault(std::shared_ptr<Allocator>);

        Allocator() = default;
        Allocator(const Allocator&) = delete;
        Allocator(Allocator&&) = delete;
        Allocator& operator=(const Allocator&) = delete;
        Allocator& operator=(Allocator&&) = delete;

        virtual ~Allocator();

        virtual uint8_t* allocate(size_t num_bytes, size_t element_size = 1) = 0;
        virtual void deallocate(uint8_t* ptr, size_t num_bytes) = 0;

        virtual void release();

        void setName(const std::string& name);
        const std::string& name() const;

        template <class T>
        T* allocate(size_t num_elems)
        {
            return ct::ptrCast<T>(allocate(num_elems * sizeof(T), sizeof(T)));
        }

        void deallocate(void* ptr, size_t size);
        template <class T>
        void deallocate(T* ptr, size_t num_elems)
        {
            deallocate(ct::ptrCast(ptr), num_elems * sizeof(T));
        }

      private:
        std::string m_name;
    };

    struct MO_EXPORTS DeviceAllocator
    {
        using Ptr_t = std::shared_ptr<DeviceAllocator>;
        using ConstPtr_t = std::shared_ptr<const DeviceAllocator>;

        static std::shared_ptr<DeviceAllocator> getDefault();

        DeviceAllocator() = default;
        DeviceAllocator(const DeviceAllocator&) = delete;
        DeviceAllocator(DeviceAllocator&&) = delete;
        DeviceAllocator& operator=(const DeviceAllocator&) = delete;
        DeviceAllocator& operator=(DeviceAllocator&&) = delete;

        virtual ~DeviceAllocator();

        virtual uint8_t* allocate(size_t num_bytes, size_t element_size = 1) = 0;
        virtual void deallocate(uint8_t* ptr, size_t num_bytes) = 0;
        void deallocate(void* ptr, size_t num_elems);

        virtual void release();

        void setName(const std::string& name);
        const std::string& name() const;

        template <class T>
        T* allocate(size_t num_elems)
        {
            return ct::ptrCast<T>(allocate(num_elems * sizeof(T), sizeof(T)));
        }

        template <class T>
        void deallocate(T* ptr, size_t num_elems)
        {
            deallocate(ct::ptrCast(ptr), num_elems * sizeof(T));
        }

      private:
        std::string m_name;
    };

} // namespace mo
