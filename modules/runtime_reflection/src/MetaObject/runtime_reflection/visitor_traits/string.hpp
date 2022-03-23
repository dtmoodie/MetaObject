#pragma once
#include "../ContainerTraits.hpp"
#include "../DynamicVisitor.hpp"

#include <string>

namespace mo
{
    template <class T>
    struct TTraits<std::basic_string<T>, 9, void> : virtual ContainerBase<std::basic_string<T>, T, void>
    {
        using base = IContainerTraits;
        void save(ISaveVisitor& visitor, const void* instance, const std::string& name, size_t) const override
        {
            auto ptr = this->ptr(instance);
            visitor(&(*ptr)[0], name, ptr->size());
        }

        void load(ILoadVisitor& visitor, void* instance, const std::string& name, size_t cnt) const override
        {
            auto ptr = this->ptr(instance);
            auto load_size = visitor.getCurrentContainerSize();
            ptr->resize(load_size);
            visitor(&(*ptr)[0], name, load_size);
        }

        void visit(StaticVisitor& visitor, const std::string& name) const
        {
            visitor.template visit<T>(name);
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

        void* valuePointer(void* inst) const override
        {
            auto ptr = this->ptr(inst);
            return &(*ptr)[0];
        }

        const void* valuePointer(const void* inst) const override
        {
            auto ptr = this->ptr(inst);
            return ptr->data();
        }
    };

} // namespace mo
