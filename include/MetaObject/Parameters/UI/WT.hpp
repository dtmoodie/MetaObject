#pragma once
#include "MetaObject/Detail/Export.hpp"
namespace mo
{
    namespace IO
    {
        namespace web
        {
            class MO_EXPORTS WebContext
            {
            public:
                static WebContext* Instance();
                void Start();
                void Stop();

            private:
                WebContext();
                struct impl;
                impl* _pimpl;
            };
        }
    }
}
