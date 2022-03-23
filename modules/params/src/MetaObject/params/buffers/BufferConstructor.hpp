#pragma once
#include "BufferFactory.hpp"

namespace mo
{
    template <class Buffer>
    struct BufferConstructor
    {
        BufferConstructor()
        {
            buffer::BufferFactory::registerConstructor(std::bind(&BufferConstructor<Buffer>::construct), Buffer::Type);
        }

        static buffer::IBuffer* construct()
        {
            return new Buffer();
        }
    };
} // namespace mo
