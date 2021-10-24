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
        void registerThread(ThreadType type, IAsyncStream::Ptr_t stream = IAsyncStream::current());
        IAsyncStream::Ptr_t getThread(ThreadType type);
        static ThreadRegistry* instance();

      private:
        ThreadRegistry();
        ~ThreadRegistry();

        struct impl;
        impl* _pimpl;
    };
} // namespace mo
#endif // MO_THREAD_REGISTRY_HPP
