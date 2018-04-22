#pragma once
#include <MetaObject/detail/Export.hpp>
#include <stddef.h>
namespace mo
{
    /*!
    * \brief The PitchedPolicy class allocates memory with padding
    *        such that a 2d array can be utilized as a texture reference
    */
    class MO_EXPORTS PitchedPolicy
    {
    public:
        PitchedPolicy();
        void sizeNeeded(int rows, int cols, int elemSize, size_t& size_needed, size_t& stride);

    private:
        size_t texture_alignment;
    };
}
