#pragma once
#include "../DynamicVisitor.hpp"
#include <memory>

namespace mo
{
    template <class T>
    struct MemoryBase : public ILoadStructTraits
    {
        using base = ILoadStructTraits;

        MemoryBase(const size_t count)
            : m_count(count)
        {
        }

        size_t size() const override
        {
            return sizeof(T);
        }

        bool triviallySerializable() const override
        {
            return false;
        }

        bool isPrimitiveType() const override
        {
            return false;
        }

        TypeInfo type() const override
        {
            return TypeInfo(typeid(T));
        }

        std::string getName() const override
        {
            return TypeInfo(typeid(T)).name();
        }

        size_t count() const override
        {
            return m_count;
        }

      private:
        size_t m_count;
    };

    template <class T>
    struct MemoryBase<const T> : public ISaveStructTraits
    {
        using base = ISaveStructTraits;

        MemoryBase(const size_t count)
            : m_count(count)
        {
        }

        size_t size() const override
        {
            return sizeof(T);
        }

        bool triviallySerializable() const override
        {
            return false;
        }

        bool isPrimitiveType() const override
        {
            return false;
        }

        TypeInfo type() const override
        {
            return TypeInfo(typeid(T));
        }

        std::string getName() const override
        {
            return TypeInfo(typeid(T)).name();
        }

        size_t count() const override
        {
            return m_count;
        }

      private:
        size_t m_count;
    };

    template <class T>
    struct TTraits<std::shared_ptr<T>, void> : public MemoryBase<T>
    {
        TTraits(std::shared_ptr<T>* ptr, const size_t count)
            : m_ptr(ptr)
            , MemoryBase<T>(count)
        {
        }

        void load(ILoadVisitor* visitor) override
        {
            size_t id = 0;

            (*visitor)(&id, "id");
            if (id != 0)
            {
                auto ptr = visitor->getPointer<T>(id);
                if (!ptr)
                {
                    *m_ptr = std::make_shared<T>();
                    (*visitor)(m_ptr->get(), "data");
                    visitor->setSerializedPointer(m_ptr->get(), id);
                    auto cache_ptr = *m_ptr;
                    visitor->pushCach(std::move(cache_ptr), std::string("shared_ptr ") + typeid(T).name(), id);
                }
                else
                {
                    auto cache_ptr =
                        visitor->popCache<std::shared_ptr<T>>(std::string("shared_ptr ") + typeid(T).name(), id);
                    if (cache_ptr)
                    {
                        *m_ptr = cache_ptr;
                    }
                }
            }
        }

        void save(ISaveVisitor* visitor) const override
        {
            size_t id = 0;
            id = size_t(m_ptr->get());
            auto ptr = visitor->getPointer<T>(id);
            (*visitor)(&id, "id");
            if (*m_ptr && ptr == nullptr)
            {
                (*visitor)(m_ptr->get(), "data");
            }
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<T>("ptr");
        }

        void* ptr() override
        {
            return nullptr;
        }

        const void* ptr() const override
        {
            return nullptr;
        }

        void setInstance(void* ptr, const TypeInfo type_) override
        {
            MO_ASSERT_EQ(type_, TypeInfo(typeid(std::shared_ptr<T>)));
            m_ptr = static_cast<std::shared_ptr<T>*>(ptr);
        }

        void setInstance(const void*, const TypeInfo) override
        {
            THROW(warn, "Trying to set const ptr");
        }

        void increment() override
        {
            ++m_ptr;
        }

      private:
        std::shared_ptr<T>* m_ptr;
    };

    template <class T>
    struct TTraits<const std::shared_ptr<T>, void> : public MemoryBase<const T>, virtual public ISaveTraits
    {
        TTraits(const std::shared_ptr<T>* ptr, const size_t count)
            : m_ptr(ptr)
            , MemoryBase<const T>(count)
        {
        }

        void save(ISaveVisitor* visitor) const override
        {
            size_t id = 0;
            id = size_t(m_ptr->get());
            auto ptr = visitor->getPointer<T>(id);
            (*visitor)(&id, "id");
            if (*m_ptr && ptr == nullptr)
            {
                (*visitor)(m_ptr->get(), "data");
            }
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<T>("ptr");
        }

        const void* ptr() const override
        {
            return nullptr;
        }

        void setInstance(const void* ptr, const TypeInfo type_) override
        {
            MO_ASSERT_EQ(type_, TypeInfo(typeid(const std::shared_ptr<T>)));
            m_ptr = static_cast<const std::shared_ptr<T>*>(ptr);
        }

        void increment() override
        {
            ++m_ptr;
        }

      private:
        const std::shared_ptr<T>* m_ptr;
    };

    template <class T>
    struct TTraits<T*, void> : public MemoryBase<T>
    {

        TTraits(T** ptr, const size_t count)
            : m_ptr(ptr)
            , MemoryBase<T>(count)
        {
        }

        void load(ILoadVisitor* visitor) override
        {
            size_t id = 0;
            // auto visitor_trait = visitor->traits();

            (*visitor)(&id, "id");
            if (id != 0)
            {
                auto ptr = visitor->getPointer<T>(id);
                if (!ptr)
                {
                    *m_ptr = new T();
                    (*visitor)(*m_ptr, "data");
                    visitor->setSerializedPointer(*m_ptr, id);
                }
                else
                {
                    *m_ptr = ptr;
                }
            }
        }

        void save(ISaveVisitor* visitor) const override
        {
            size_t id = 0;
            // auto visitor_trait = visitor->traits();

            id = size_t(*m_ptr);
            auto ptr = visitor->getPointer<T>(id);
            (*visitor)(&id, "id");
            if (*m_ptr && ptr == nullptr)
            {
                (*visitor)(*m_ptr, "data");
            }
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<T>("ptr");
        }

        void* ptr() override
        {
            return nullptr;
        }
        void* ptr() const override
        {
            return nullptr;
        }
        void increment() const override
        {
            ++m_ptr;
        }
        void setInstance(void* ptr, const TypeInfo type_) override
        {
            MO_ASSERT_EQ(type_, TypeInfo(typeid(T*)));
            m_ptr = ptr;
        }
        void setInstance(const void*, const TypeInfo) override
        {
            THROW(warn, "Trying to set const ptr");
        }

      private:
        T** m_ptr;
    };

    template <class T>
    struct TTraits<const T*, void> : public MemoryBase<const T>
    {

        TTraits(const T** ptr, const size_t count)
            : m_ptr(ptr)
            , MemoryBase<const T>(count)
        {
        }

        void visit(ILoadVisitor* visitor) const override
        {
            size_t id = 0;
            // auto visitor_trait = visitor->traits();

            id = size_t(*m_ptr);
            auto ptr = visitor->getPointer<T>(id);
            (*visitor)(&id, "id");
            if (*m_ptr && ptr == nullptr)
            {
                (*visitor)(*m_ptr, "data");
            }
        }

        void visit(StaticVisitor* visitor) const override
        {
            visitor->template visit<T>("ptr");
        }

        void* ptr() override
        {
            return nullptr;
        }

        void* ptr() const override
        {
            return nullptr;
        }

        void increment() const override
        {
            ++m_ptr;
        }

        void setInstance(void* ptr, const TypeInfo type_) override
        {
            MO_ASSERT_EQ(type_, TypeInfo(typeid(const T*)));
            m_ptr = ptr;
        }
        void setInstance(const void*, const TypeInfo) override
        {
            THROW(warn, "Trying to set const ptr");
        }

      private:
        const T** m_ptr;
    };
}
