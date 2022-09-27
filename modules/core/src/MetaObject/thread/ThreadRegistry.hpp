#ifndef MO_THREAD_REGISTRY_HPP
#define MO_THREAD_REGISTRY_HPP

#include <MetaObject/core.hpp>
#include <cstddef>

namespace mo
{
    size_t MO_EXPORTS getThisThread();
    class MO_EXPORTS ThreadRegistry
    {
      public:
        enum ThreadType
        {
            GUI,
            ANY,
            WORKER
        };
        virtual ~ThreadRegistry();

        virtual IAsyncStreamPtr_t getGUIStream() = 0;
        virtual void setGUIStream(IAsyncStreamPtr_t) = 0;

        virtual void registerThread(ThreadType type, IAsyncStreamPtr_t stream = IAsyncStream::current()) = 0;
        virtual IAsyncStreamPtr_t getThread(ThreadType type) = 0;

        static ThreadRegistry* instance();
        static ThreadRegistry* instance(SystemTable* table);
    };
} // namespace mo
#endif // MO_THREAD_REGISTRY_HPP
