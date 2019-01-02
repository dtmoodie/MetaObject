#pragma once
#include "../DynamicVisitor.hpp"
#include <memory>

namespace mo
{
    template<class T>
    struct MemoryBase : public IStructTraits
    {
        using base = IStructTraits;

        virtual size_t size() const override { return sizeof(T); }
        virtual bool triviallySerializable() const override { return false; }
        virtual bool isPrimitiveType() const override { return false; }
        virtual TypeInfo type() const override { return TypeInfo(typeid(T)); }
        virtual std::string getName() const { return TypeInfo(typeid(T)).name(); }
    };

    template <class T>
    struct TTraits<std::shared_ptr<T>, void> : public MemoryBase<T>
    {
        TTraits(std::shared_ptr<T>* ptr) : m_ptr(ptr){}

        void visit(IReadVisitor* visitor) override
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

        void visit(IWriteVisitor* visitor) const override
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

        void* ptr() override{return nullptr;}
        const void* ptr() const override{return nullptr;}

      private:
        std::shared_ptr<T>* m_ptr;
    };

    template <class T>
    struct TTraits<const std::shared_ptr<T>, void> : public MemoryBase<T>
    {
        TTraits(const std::shared_ptr<T>* ptr) : m_ptr(ptr){}

        void visit(IReadVisitor* ) override
        {
            throw std::runtime_error("Tried to read into a const ptr");
        }

        virtual void visit(IWriteVisitor* visitor) const override
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

        void* ptr() override{return nullptr;}
        const void* ptr() const override{return nullptr;}

      private:
        const std::shared_ptr<T>* m_ptr;
    };

    template <class T>
    struct TTraits<T*, void> : public MemoryBase<T>
    {

        TTraits(T** ptr) : m_ptr(ptr){}

        virtual void visit(IReadVisitor* visitor) override
        {
            size_t id = 0;
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
            size_t id = 0;
            auto visitor_trait = visitor->traits();

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

        void* ptr() override{return nullptr;}
        void* ptr() const override{return nullptr;}
      private:
        T** m_ptr;
    };

    template <class T>
    struct TTraits<const T*, void> : public MemoryBase<T>
    {

        TTraits(T** ptr) : m_ptr(ptr){}

        virtual void visit(IReadVisitor* visitor) override
        {
            throw std::runtime_error("Tried to read data into a const ptr");
        }

        virtual void visit(IWriteVisitor* visitor) const override
        {
            size_t id = 0;
            auto visitor_trait = visitor->traits();

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

        void* ptr() override{return nullptr;}
        void* ptr() const override{return nullptr;}
      private:
        T** m_ptr;
    };
}
