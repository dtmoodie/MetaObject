#pragma once
#include "MetaObject/types.hpp"
#include "small_vec_storage.hpp"
#include <vector>

namespace mo
{
    template <class T, int N>
    struct SmallVec : public SmallVecBase<T>, public SmallVecStorage<T, N>
    {
        SmallVec() : SmallVecStorage<T, N>(*static_cast<SmallVecBase<T>*>(this)) {}

        SmallVec(const SmallVecBase<T>& other) : SmallVecStorage<T, N>(*static_cast<SmallVecBase<T>*>(this))
        {
            SmallVecStorage<T, N>::assign(other.begin(), other.end());
        }

        SmallVec(const std::vector<T>& vec) : SmallVecStorage<T, N>(*static_cast<SmallVecBase<T>*>(this))
        {
            const size_t size = vec.size();
            if (size)
            {
                SmallVecStorage<T, N>::assign(&vec[0], &vec[0] + size);
            }
        }
        ~SmallVec()
        {
            SmallVecStorage<T, N>::~SmallVecStorage();
            SmallVecBase<T>::~SmallVecBase();
        }

        SmallVec<T, N>& operator=(const std::vector<T>& vec)
        {
            const size_t size = vec.size();
            if (size)
            {
                SmallVecStorage<T, N>::assign(&vec[0], &vec[0] + size);
            }
            return *this;
        }
    };
}
