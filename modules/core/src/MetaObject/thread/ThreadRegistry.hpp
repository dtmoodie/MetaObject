#pragma once
#include "MetaObject/detail/Export.hpp"
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
        void registerThread(ThreadType type, size_t id = getThisThread());
        size_t getThread(int type);
        static ThreadRegistry* instance();

      private:
        ThreadRegistry();
        ~ThreadRegistry();

        struct impl;
        impl* _pimpl;
    };
}
