#pragma once
#include "small_vec_storage.hpp"
#include <vector>

namespace mo
{
    template<class T, int N>
    struct SmallVec : public SmallVecBase<T>, public SmallVecStorage<T, N>
    {
        SmallVec() :
            SmallVecStorage<T, N>(*static_cast<SmallVecBase<T>*>(this))
        {

        }

        SmallVec(const std::vector<T>& vec):
            SmallVecStorage<T, N>(*static_cast<SmallVecBase<T>*>(this))
        {
            const size_t size = vec.size();
            if (size)
            {
                assign(&vec[0], &vec[0] + size);
            }
        }

        SmallVec<T, N>& operator=(const std::vector<T>& vec)
        {
            const size_t size = vec.size();
            if (size)
            {
                assign(&vec[0], &vec[0] + size);
            }
            return *this;
        }
    };
}