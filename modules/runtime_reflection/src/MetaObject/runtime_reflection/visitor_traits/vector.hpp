#pragma once
#include "../ContainerTraits.hpp"
#include "../DynamicVisitor.hpp"

namespace mo
{
    template <class T, class A>
    struct TTraits<std::vector<T, A>, 4, void>: virtual ContainerBase<std::vector<T, A>, T>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string& name, const size_t) const override
        {
            auto val = this->ptr(inst);
            visitor(val->data(), name, val->size());
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string& name, const size_t) const override
        {
            auto val = this->ptr(inst);
            visitor(val->data(), name, val->size());
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T>("data");
        }

        size_t getContainerSize(const void* inst) const override
        {
            auto ptr = this->ptr(inst);
            return ptr->size();
        }

        void setContainerSize(size_t size, void* inst) const override
        {
            auto ptr = this->ptr(inst);
            ptr->resize(size);
        }

        void* valuePointer(void* inst) const
        {
            auto val = this->ptr(inst);
            return val->data();
        }

        const void* valuePointer(const void* inst) const
        {
            auto val = this->ptr(inst);
            return val->data();
        }
    };
}
