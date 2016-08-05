/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/parameters
*/
#pragma once

#ifdef PARAMETERS_GENERATE_UI
#ifdef Qt5_FOUND
#include "parameters/UI/Qt/ParameterProxy.hpp"
#endif
#endif

namespace Parameters
{
    namespace UI
    {
#ifdef PARAMETERS_GENERATE_UI
        template<typename T> class UiPolicy
        {
        public:
            UiPolicy()
            {
#ifdef Qt5_FOUND
                static qt::QtUiPolicy<T> qt_policy;
#endif
            }
        };
#else
        class NoUiPolicy
        {        };
        template<typename T> class UiPolicy: public NoUiPolicy
        {
        public:

        };
#endif

    }
}
