#ifndef MO_SIGNALS_ARGUMENTPACK_HPP
#define MO_SIGNALS_ARGUMENTPACK_HPP
#include <MetaObject/signals/ArgumentSyncPolicy.hpp>

#include <tuple>
namespace mo
{
    template <class... ARGS>
    struct ArgumentPack
    {
        ArgumentPack(IAsyncStream& src, IAsyncStream& dst, ARGS&&... args)
            : data(ArgumentSyncPolicy<typename std::decay<ARGS>::type>::syncToStream(args, src, dst)...)
        {
        }

        std::tuple<typename std::decay<ARGS>::type...> data;
    };
} // namespace mo

#endif // MO_SIGNALS_ARGUMENTPACK_HPP