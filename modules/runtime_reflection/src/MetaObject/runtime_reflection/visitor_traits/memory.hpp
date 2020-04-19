#pragma once
#include "../DynamicVisitor.hpp"
#include "../StructTraits.hpp"

#include <MetaObject/logging/logging.hpp>

//#include <ce/shared_ptr.hpp>
#include <memory>

namespace mo
{
    template <class T>
    struct PointerWrapper
    {
        std::shared_ptr<T> ptr;
    };
    template <class T>
    void loadPointer(ILoadVisitor& visitor, PointerWrapper<T>* val)
    {
        uint32_t id = 0;
        visitor(&id, "id");
        id = id & (~0x80000000);
        if (id != 0)
        {
            auto ptr = visitor.getPointer<T>(id);
            if (!ptr)
            {
                val->ptr = std::make_shared<T>();
                visitor(val->ptr.get(), "data");
                visitor.setSerializedPointer(val->ptr.get(), id);
                std::shared_ptr<T> cache_ptr = val->ptr;
                visitor.pushCach(std::move(cache_ptr), std::string("shared_ptr ") + typeid(T).name(), id);
            }
            else
            {
                auto cache_ptr =
                    visitor.popCache<std::shared_ptr<T>>(std::string("shared_ptr ") + typeid(T).name(), id);
                if (cache_ptr)
                {
                    val->ptr = cache_ptr;
                }
            }
        }
    }

    template <class T>
    void savePointer(ISaveVisitor& visitor, const PointerWrapper<T>* val)
    {
        uint32_t id = visitor.getPointerId(TypeInfo::create<std::shared_ptr<T>>(), ct::ptrCast<void>(val));
        auto ptr = visitor.getPointer<T>(id);
        visitor(&id, "id");
        if (val->ptr && ptr == nullptr)
        {
            visitor(val->ptr.get(), "data");
        }
    }

    template <class T>
    struct TTraits<PointerWrapper<T>, 4> : virtual StructBase<PointerWrapper<T>>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override
        {
            loadPointer(visitor, this->ptr(inst));
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            auto val = this->ptr(inst);
            savePointer(visitor, val);
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<size_t>("id");
            visitor.template visit<T>("ptr");
        }
    };

    template <class T>
    struct TTraits<PointerWrapper<const T>, 4> : virtual StructBase<PointerWrapper<T>>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override
        {
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            auto val = this->ptr(inst);
            savePointer(visitor, val);
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<size_t>("id");
            visitor.template visit<T>("ptr");
        }
    };

    template <class T>
    void loadPointer(ILoadVisitor& visitor, std::shared_ptr<T>& val)
    {
        uint32_t id = 0;
        visitor(&id, "id");
        id = id & (~0x80000000);
        if (id != 0)
        {
            auto ptr = visitor.getPointer<T>(id);
            if (!ptr)
            {
                val = std::make_shared<T>();
                visitor(val.get(), "data");
                visitor.setSerializedPointer(val.get(), id);
                auto cache_ptr = val;
                visitor.pushCach(std::move(cache_ptr), std::string("shared_ptr ") + typeid(T).name(), id);
            }
            else
            {
                std::shared_ptr<T> cache_ptr;
                const auto success = visitor.tryPopCache(cache_ptr, std::string("shared_ptr ") + typeid(T).name(), id);
                if (success && cache_ptr)
                {
                    val = cache_ptr;
                }
            }
        }
    }

    template <class T>
    void savePointer(ISaveVisitor& visitor, const std::shared_ptr<T>& val)
    {
        uint32_t id = visitor.getPointerId(TypeInfo::create<typename std::remove_const<T>::type>(), val.get());

        auto ptr = visitor.getPointer<typename std::remove_const<T>::type>(id);
        visitor(&id, "id");
        if (val && ptr == nullptr)
        {
            visitor(val.get(), "data");
            visitor.setSerializedPointer(val.get(), id);
        }
    }

    template <class T>
    struct TTraits<std::shared_ptr<T>, 4> : virtual StructBase<std::shared_ptr<T>>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override
        {
            const auto traits = visitor.traits();
            // This is effectively just checking if we are serializing to a cereal json archive or a binary archive
            if (traits.supports_named_access)
            {
                PointerWrapper<T> pointer_wrapper;
                visitor(&pointer_wrapper, "ptr_wrapper");
                *this->ptr(inst) = std::move(pointer_wrapper.ptr);
            }
            else
            {
                auto val = this->ptr(inst);
                loadPointer(visitor, *val);
            }
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            const auto traits = visitor.traits();
            if (traits.supports_named_access)
            {
                PointerWrapper<T> wrapper{*this->ptr(inst)};
                visitor(&wrapper, "ptr_wrapper");
            }
            else
            {
                MO_ASSERT_EQ(cnt, 1);
                auto val = this->ptr(inst);
                savePointer(visitor, *val);
            }
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T>("ptr");
        }
    };

    template <class T>
    struct TTraits<std::shared_ptr<const T>, 4> : virtual StructBase<std::shared_ptr<T>>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override
        {
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            const auto traits = visitor.traits();
            if (traits.supports_named_access)
            {
                PointerWrapper<const T> wrapper{this->ref(inst)};
                visitor(&wrapper, "ptr_wrapper");
            }
            else
            {
                MO_ASSERT_EQ(cnt, 1);
                savePointer(visitor, this->ref(inst));
            }
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
