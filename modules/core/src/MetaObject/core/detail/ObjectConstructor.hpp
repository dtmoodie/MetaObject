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

        SharedPtr_t makeShared() const
        {
            return SharedPtr_t(create());
        }

        SharedPtr_t createShared() const
        {
            return makeShared();
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
} // namespace mo

#endif // MO_CORE_OBJECT_CONSTRUCTOR_HPP
