#pragma once
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/signals/Connection.hpp"
//#include "MetaObject/thread/InterThread.hpp"
#include <MetaObject/core/Context.hpp>
namespace mo
{
    template <class Sig, class Mutex>
    class TSignalRelay;
}
