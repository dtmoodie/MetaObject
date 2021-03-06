#ifndef MO_SIGNALS_ARGUMENTSYNCPOLICY_HPP
#define MO_SIGNALS_ARGUMENTSYNCPOLICY_HPP
#include <MetaObject/core/IAsyncStream.hpp>

namespace mo
{
    template <class T>
    struct ArgumentSyncPolicy
    {
        using type = T;

        static type syncToStream(const type& data, IAsyncStream& src, IAsyncStream& dst)
        {
            return data;
        }
    };

    template <class T>
    using ArgumentSyncPolicy_t = typename ArgumentSyncPolicy<T>::type;

} // namespace mo

#endif // MO_SIGNALS_ARGUMENTSYNCPOLICY_HPP