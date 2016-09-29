#include "MetaObject/Context.hpp"
#include "MetaObject/Thread/ThreadRegistry.hpp"
using namespace mo;

Context::Context()
{
    thread_id = GetThisThread();
}