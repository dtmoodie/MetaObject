#ifndef MO_METAPARAMS_COMMON_HPP
#define MO_METAPARAMS_COMMON_HPP
#include <MetaObject/runtime_reflection/TraitInterface.hpp>

namespace mo
{
    namespace metaparams
    {
        template <class T>
        void registerTrait()
        {
            mo::TTraits<T> trait;
        }
    } // namespace metaparams
} // namespace mo

#endif // MO_METAPARAMS_COMMON_HPP