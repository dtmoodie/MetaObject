#pragma once

namespace mo
{
    template<class T, int N, typename Enable = void> struct MetaParameter;

#define MO_METAPARAMTER_SERIALIZATION_POLICY(N) \
template<class T> struct MetaParameter<T, N>: public MetaParameter<T, N-1, void> \
{ \
    
    
}
}