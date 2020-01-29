#ifndef MO_VISITATION_ARRAY_ADAPTER_HPP
#define MO_VISITATION_ARRAY_ADAPTER_HPP
#include "../ContainerTraits.hpp"

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/runtime_reflection/IDynamicVisitor.hpp>
#include <MetaObject/types/ArrayAdapater.hpp>

#include <ct/types/TArrayView.hpp>
namespace mo
{
    template <class T, size_t N>
    struct TTraits<std::array<T, N>, 4, void> : virtual ContainerBase<std::array<T, N>, T>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override
        {
            visitor(this->ptr(inst)->data(), "data", N);
        }

        void save(ISaveVisitor& visitor, const void* val, const std::string&, size_t) const override
        {
            visitor(this->ptr(val)->data(), "data", N);
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T>("data", N);
        }

        bool triviallySerializable() const
        {
            return true;
        }

        bool isContinuous() const override
        {
            return true;
        }

        size_t getContainerSize(const void*) const override
        {
            return N;
        }

        void setContainerSize(size_t size, void*) const override
        {
            MO_ASSERT_EQ(N, size);
        }

        void* valuePointer(void* inst) const override
        {
            return static_cast<std::array<T, N>*>(inst)->data();
        }

        const void* valuePointer(const void* inst) const override
        {
            return static_cast<const std::array<T, N>*>(inst)->data();
        }
    };

    template <class T>
    struct TTraits<ct::TArrayView<T>, 4, void> : virtual ContainerBase<ct::TArrayView<T>, T>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override
        {
            auto ptr = this->ptr(inst);
            visitor(ptr->data(), "data", ptr->size());
        }

        void save(ISaveVisitor& visitor, const void* val, const std::string&, size_t) const override
        {
            auto ptr = this->ptr(val);
            visitor(ptr->data(), "data", ptr->size());
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T>("data");
        }

        bool triviallySerializable() const
        {
            return false;
        }

        bool isContinuous() const override
        {
            return true;
        }

        size_t getContainerSize(const void* inst) const override
        {
            auto ptr = this->ptr(inst);
            return ptr->size();
        }

        void setContainerSize(size_t size, void* inst) const override
        {
            auto ptr = this->ptr(inst);
            MO_ASSERT_EQ(ptr->size(), size);
        }

        void* valuePointer(void* inst) const override
        {
            return this->ptr(inst)->data();
        }

        const void* valuePointer(const void* inst) const override
        {
            return this->ptr(inst)->data();
        }
    };

    template <class T>
    struct TTraits<ct::TArrayView<const T>, 4, void> : virtual ContainerBase<ct::TArrayView<const T>, const T>
    {
        void load(ILoadVisitor&, void* , const std::string&, size_t) const override
        {
            THROW(warn, "Unable to load to a const array view");
        }

        void save(ISaveVisitor& visitor, const void* val, const std::string&, size_t) const override
        {
            auto ptr = this->ptr(val);
            visitor(ptr->data(), "data", ptr->size());
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T>("data");
        }

        bool triviallySerializable() const
        {
            return false;
        }

        bool isContinuous() const override
        {
            return true;
        }

        size_t getContainerSize(const void* inst) const override
        {
            auto ptr = this->ptr(inst);
            return ptr->size();
        }

        void setContainerSize(size_t size, void* inst) const override
        {
            auto ptr = this->ptr(inst);
            MO_ASSERT_EQ(ptr->size(), size);
        }

        void* valuePointer(void*) const override
        {
            return nullptr;
        }

        const void* valuePointer(const void* inst) const override
        {
            return this->ptr(inst)->data();
        }
    };

    template <class T, size_t ROWS, size_t COLS>
    struct Visit<MatrixAdapter<T, ROWS, COLS>>
    {
        static ILoadVisitor&
        load(ILoadVisitor& visitor, MatrixAdapter<T, ROWS, COLS>* val, const std::string&, const size_t)
        {
            visitor(val->ptr, "data", ROWS * COLS);
            return visitor;
        }

        static ISaveVisitor&
        save(ISaveVisitor& visitor, const MatrixAdapter<T, ROWS, COLS>* val, const std::string&, const size_t)
        {
            visitor(val->ptr, "data", ROWS * COLS);
            return visitor;
        }

        static void visit(StaticVisitor& visitor, const std::string&, const size_t)
        {
            visitor.template visit<T>("data", ROWS * COLS);
        }
    };
}

#endif // MO_VISITATION_ARRAY_ADAPTER_HPP
