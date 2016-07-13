#pragma once
#include "MetaObject/Detail/Export.hpp"

namespace mo
{
    size_t MO_EXPORTS GetThisThread();
    class MO_EXPORTS ThreadRegistry
    {
    public:
        virtual void register_thread(int type, size_t id = GetThisThread());
        virtual size_t get_thread(int type);

        static ThreadRegistry* get_instance();

    private:
        
        ThreadRegistry();
        ~ThreadRegistry();
        
        struct impl;
        impl* _pimpl;
    };
}
