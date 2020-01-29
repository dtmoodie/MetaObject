#ifndef MO_CORE_OBJECT_CONSTRUCTOR_HPP
#define MO_CORE_OBJECT_CONSTRUCTOR_HPP
#include <memory>

namespace mo
{
    template <class T>
    struct ObjectConstructor
    {
        using SharedPtr_t = std::shared_ptr<T>;
        using UniquePtr_t = std::unique_ptr<T>;

        SharedPtr_t createShared() const
        {
            return SharedPtr_t(create());
        }

        UniquePtr_t createUnique() const
        {
            return UniquePtr_t(create());
        }

        T* create() const
        {
            return new T();
        }
    };
}

#endif // MO_CORE_OBJECT_CONSTRUCTOR_HPP
