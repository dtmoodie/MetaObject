#pragma once
#include "MetaObject/detail/Export.hpp"

namespace std
{
    template<class T>
    class function;
}

namespace mo
{
    namespace python
    {
        void MO_EXPORTS pythonSetup(const char* module_name);
        void MO_EXPORTS registerSetupFunction(std::function<void(void)>&& func);
    }
}