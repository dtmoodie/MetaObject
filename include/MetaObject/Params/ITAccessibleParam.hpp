#pragma once
#include "ITParam.hpp"

namespace mo{
template<typename T> class ITAccessibleParam: virtual public ITParam<T>{
public:
    virtual AccessToken<T> access() = 0;
};

}