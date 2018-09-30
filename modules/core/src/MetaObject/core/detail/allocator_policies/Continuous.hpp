#pragma once
#include <MetaObject/detail/Export.hpp>
#include <stddef.h>
namespace mo
{
    /*!
    * \brief The ContinuousPolicy class allocates memory with zero padding
    *        wich allows for nice reshaping operations
    *
    */
    class MO_EXPORTS ContinuousPolicy
    {
      public:
        static void sizeNeeded(int rows, int cols, int elemSize, size_t& size_needed, size_t& stride);
    };
}
