#pragma once
#include "Defs.h"

#include <boost/thread.hpp>
namespace Signals
{
    size_t SIGNAL_EXPORTS get_thread_id(const boost::thread::id& id);
}
