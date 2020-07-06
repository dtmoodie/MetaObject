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
            ANY
        };
        void registerThread(ThreadType type, mo::IAsyncStreamPtr_t stream = mo::IAsyncStream::current());
        mo::IAsyncStreamPtr_t getThread(int type);
        static ThreadRegistry* instance();

      private:
        ThreadRegistry();
        ~ThreadRegistry();

        struct impl;
        impl* _pimpl;
    };
} // namespace mo
#endif // MO_THREAD_REGISTRY_HPP
