#pragma once
#include "BufferFactory.hpp"

namespace mo
{
    template <class T>
    class BufferConstructor
    {
      public:
        BufferConstructor()
        {
            static_assert(T::Type != TParam_e, "T Param not a buffer");
            Buffer::BufferFactory::RegisterFunction(TypeInfo(typeid(typename T::ValueType)),
                                                    std::bind(&BufferConstructor<T>::create, std::placeholders::_1),
                                                    T::Type);
        }
        static IParam* create(IParam* input)
        {
            T* ptr = new T();
            if (ptr->setInput(input))
            {
                return ptr;
            }
            delete ptr;
            return nullptr;
        }
    };
}