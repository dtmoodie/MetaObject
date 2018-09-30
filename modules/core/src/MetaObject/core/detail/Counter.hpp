#pragma once
#include <stdint.h>

namespace mo
{
    template <int32_t N>
    struct _counter_
    {
        _counter_<N - 1> operator--() { return _counter_<N - 1>(); }

        _counter_<N + 1> operator++() { return _counter_<N + 1>(); }

        operator int32_t(){return N;}
    };
}
