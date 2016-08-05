#pragma once

#include "BufferFactory.hpp"

namespace mo
{
    template<class T, Buffer::BufferFactory::buffer_type type> class BufferConstructor
    {
    public:
        BufferConstructor()
        {
            Buffer::BufferFactory::RegisterFunction(
                TypeInfo(typeid(typename T::ParameterType)),
                std::bind(&BufferConstructor<T, type>::create, std::placeholders::_1), 
                type);
        }
        static IParameter* create(IParameter* input)
        {
            T* ptr = new T();
            if(ptr->SetInput(input))
            {
                return ptr;
            }
            delete ptr;
            return nullptr;
        }
    };
}