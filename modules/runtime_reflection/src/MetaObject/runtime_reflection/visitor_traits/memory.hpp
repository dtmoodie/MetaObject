#pragma once
#include "../DynamicVisitor.hpp"
#include "../StructTraits.hpp"

#include <MetaObject/core/detail/ObjectConstructor.hpp>
#include <MetaObject/logging/logging.hpp>

#include <memory>

namespace mo
{
    template <class T>
    struct PointerWrapper
    {
        using element_type = T;
        PointerWrapper& operator=(const std::shared_ptr<T>& v)
        {
            ptr = v;
            return *this;
        }
        T* get()
        {
            return ptr.get();
        }

        const T* get() const
        {
            return ptr.get();
        }

        operator bool() const
        {
            return ptr.operator bool();
        }

        operator std::shared_ptr<T>() const
        {
            return ptr;
        }
        std::shared_ptr<T> ptr;
    };

    template <class T>
    bool isWrapper(const T&)
    {
        return false;
    }
    template <class T>
    bool isWrapper(const PointerWrapper<T>&)
    {
        return true;
    }

    template <class T>
    struct PolymorphicSerializationHelper
    {
        template <class Ptr_t>
        static void load(ILoadVisitor&, Ptr_t& ptr)
        {
            ObjectConstructor<T> ctr;
            ptr = ctr.makeShared();
        }
        template <class Ptr_t>
        static void save(ISaveVisitor&, const Ptr_t&)
        {
        }
    };

    template <class Ptr_t>
    void loadPointer(ILoadVisitor& visitor, Ptr_t& val)
    {
        using T = typename Ptr_t::element_type;

        const auto traits = visitor.traits();
        // This is effectively just checking if we are serializing to a cereal json archive or a binary archive
        if (traits.supports_named_access && !isWrapper(val))
        {
            PointerWrapper<T> pointer_wrapper;
            visitor(&pointer_wrapper, "ptr_wrapper");
            val = std::move(pointer_wrapper.ptr);
        }
        else
        {
            uint32_t id = 0;
            visitor(&id, "id");
            id = id & (~0x80000000);
            if (id != 0)
            {
                auto ptr = visitor.getPointer<T>(id);
                if (!ptr)
                {
                    PolymorphicSerializationHelper<T>::load(visitor, val);
                    // val = std::make_shared<T>();
                    visitor(val.get(), "data");
                    visitor.setSerializedPointer(val.get(), id);
                    std::shared_ptr<T> cache_ptr = val;
                    visitor.pushCach(std::move(cache_ptr), std::string("shared_ptr ") + typeid(T).name(), id);
                }
                else
                {
                    std::shared_ptr<T> cache_ptr;
                    const auto success =
                        visitor.tryPopCache(cache_ptr, std::string("shared_ptr ") + typeid(T).name(), id);
                    if (success && cache_ptr)
                    {
                        val = cache_ptr;
                    }
                }
            }
        }
    }

    template <class Ptr_t>
    void savePointer(ISaveVisitor& visitor, const Ptr_t& val)
    {
        using T = typename Ptr_t::element_type;

        const auto traits = visitor.traits();
        if (traits.supports_named_access && !isWrapper(val))
        {
            PointerWrapper<T> wrapper{val};
            visitor(&wrapper, "ptr_wrapper");
        }
        else
        {
            uint32_t id = visitor.getPointerId(TypeInfo::create<typename std::remove_const<T>::type>(), val.get());
            auto ptr = visitor.getPointer<typename std::remove_const<T>::type>(id);
            visitor(&id, "id");
            if (val && ptr == nullptr)
            {
                PolymorphicSerializationHelper<T>::save(visitor, val);
                visitor(val.get(), "data");
                visitor.setSerializedPointer(val.get(), id);
            }
        }
    }
    template <class T>
    struct SharedPointerHelper
    {
        template <class Ptr_t>
        static void load(ILoadVisitor& visitor, Ptr_t& val)
        {
            loadPointer(visitor, val);
        }
        template <class Ptr_t>
        static void save(ISaveVisitor& visitor, const Ptr_t& val)
        {
            savePointer(visitor, val);
        }
    };

    template <class T>
    struct SharedPointerHelper<const T>
    {
        template <class Ptr_t>
        static void load(ILoadVisitor& visitor, Ptr_t& val)
        {
        }
        template <class Ptr_t>
        static void save(ISaveVisitor& visitor, const Ptr_t& val)
        {
            savePointer(visitor, val);
        }
    };

    template <class T>
    struct TTraits<PointerWrapper<T>, 4> : virtual StructBase<PointerWrapper<T>>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            auto& ref = this->ref(inst);
            SharedPointerHelper<T>::load(visitor, ref);
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            auto& ref = this->ref(inst);
            SharedPointerHelper<T>::save(visitor, ref);
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<size_t>("id");
            visitor.template visit<T>("ptr");
        }
    };

    template <class T>
    struct TTraits<std::shared_ptr<T>, 4> : virtual StructBase<std::shared_ptr<T>>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            auto& ref = this->ref(inst);
            SharedPointerHelper<T>::load(visitor, ref);
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            auto& ref = this->ref(inst);
            SharedPointerHelper<T>::save(visitor, ref);
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T>("ptr");
        }
    };

    template <class T>
    struct TTraits<T*, 3> : virtual StructBase<T*>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            size_t id = 0;
            auto m_ptr = this->ptr(inst);
            visitor(&id, "id");
            if (id != 0)
            {
                auto ptr = visitor.getPointer<T>(id);
                if (!ptr)
                {
                    *m_ptr = new T();
                    visitor(*m_ptr, "data");
                    visitor.setSerializedPointer(*m_ptr, id);
                }
                else
                {
                    *m_ptr = ptr;
                }
            }
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            size_t id = 0;
            // auto visitor_trait = visitor->traits();
            auto m_ptr = this->ptr(inst);
            id = size_t(*m_ptr);
            auto ptr = visitor.getPointer<T>(id);
            visitor(&id, "id");
            if (*m_ptr && ptr == nullptr)
            {
                visitor(*m_ptr, "data");
            }
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T>("ptr");
        }
    };
} // namespace mo
