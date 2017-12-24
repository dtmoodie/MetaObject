#pragma once
#include "MetaObject/detail/Export.hpp"
#include <boost/python/object_fwd.hpp>
#include <cstdint>
#include <vector>
#include <functional>

class IObjectConstructor;

namespace mo
{
    namespace python
    {
        void MO_EXPORTS pythonSetup(const char* module_name);
        void MO_EXPORTS registerSetupFunction(std::function<void(void)>&& func);
        void MO_EXPORTS registerInterfaceSetupFunction(uint32_t interface_id, std::function<void(std::vector<IObjectConstructor*>&)>&& func);
        void MO_EXPORTS registerObjects();
        template<class T>
        struct RegisterInterface
        {
            RegisterInterface(void(*setup)(), void(*construct)(std::vector<IObjectConstructor*>&))
            {
                registerSetupFunction(setup);
                registerInterfaceSetupFunction(T::s_interfaceID, construct);
            }
        };
    }
}
