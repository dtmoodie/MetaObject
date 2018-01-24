#pragma once
#include "small_vec_storage.hpp"

namespace mo
{
    template<class T, class N>
    struct SmallVec : public SmallVecBase, public SmallVecStorage
    {
        SmallVec() :
            SmallVecStorage(*static_cast<SmallVecBase*>(this))
        {

        }

        SmallVec(const std::vector<T>& vec)
            SmallVec()
        {
            const size_t size = vec.size();
            if (size)
            {
                assign(&vec[0], &vec[0] + size);
            }
        }

    };
}