#pragma once
#include "../DynamicVisitor.hpp"
#include <memory>

namespace mo
{
    template <class T>
    struct TTraits<std::shared_ptr<T>, void> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(std::shared_ptr<T>* ptr, const std::shared_ptr<T>* const_ptr) : m_ptr(ptr), m_const_ptr(const_ptr) {}
        virtual void visit(IReadVisitor* visitor) override
        {
            uint64_t id = 0;

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

        virtual void visit(IWriteVisitor* visitor) const override
        {
            uint64_t id = 0;
            if (m_const_ptr)
            {
                id = uint64_t(m_const_ptr->get());
                auto ptr = visitor->getPointer<T>(id);
                (*visitor)(&id, "id");
                if (*m_const_ptr && ptr == nullptr)
                {
                    (*visitor)(m_const_ptr->get(), "data");
                }
            }
            else
            {
                id = uint64_t(m_ptr->get());
                auto ptr = visitor->getPointer<T>(id);
                (*visitor)(&id, "id");
                if (*m_ptr && ptr == nullptr)
                {
                    (*visitor)(m_ptr->get(), "data");
                }
            }
        }

        virtual size_t size() const override { return sizeof(T*); }
        virtual bool triviallySerializable() const override { return false; }
        virtual bool isPrimitiveType() const override { return false; }
        virtual TypeInfo type() const override { return TypeInfo(typeid(T)); }
        virtual const void* ptr() const override { return nullptr; }
        virtual void* ptr() override { return nullptr; }
        virtual const char* getName() const { return typeid(std::shared_ptr<T>).name(); }
      private:
        std::shared_ptr<T>* m_ptr;
        const std::shared_ptr<T>* m_const_ptr;
    };

    template <class T>
    struct TTraits<T*, void> : public IStructTraits
    {
        using base = IStructTraits;

        TTraits(T** ptr, T* const* const_ptr) : m_ptr(ptr), m_const_ptr(const_ptr) {}

        virtual void visit(IReadVisitor* visitor) override
        {
            uint64_t id = 0;
            auto visitor_trait = visitor->traits();

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

        virtual void visit(IWriteVisitor* visitor) const override
        {
            if (m_const_ptr)
            {
                uint64_t id = 0;
                auto visitor_trait = visitor->traits();

                id = uint64_t(*m_const_ptr);
                auto ptr = visitor->getPointer<T>(id);
                (*visitor)(&id, "id");
                if (*m_const_ptr && ptr == nullptr)
                {
                    (*visitor)(*m_const_ptr, "data");
                }
            }
            else
            {
                uint64_t id = 0;
                auto visitor_trait = visitor->traits();

                id = uint64_t(*m_ptr);
                auto ptr = visitor->getPointer<T>(id);
                (*visitor)(&id, "id");
                if (*m_ptr && ptr == nullptr)
                {
                    (*visitor)(*m_ptr, "data");
                }
            }
        }

        virtual size_t size() const override { return sizeof(T*); }
        virtual bool triviallySerializable() const override { return false; }
        virtual bool isPrimitiveType() const override { return false; }
        virtual TypeInfo type() const override { return TypeInfo(typeid(T)); }
        virtual const void* ptr() const override { return nullptr; }
        virtual void* ptr() override { return nullptr; }
        virtual const char* getName() const { return typeid(T*).name(); }
      private:
        T** m_ptr;
        T* const* m_const_ptr;
    };
}
