#pragma once
#include "MetaObject/detail/Export.hpp"
#include <boost/python/object_fwd.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

struct IObjectConstructor;
class SystemTable;

namespace mo
{
    namespace python
    {
        std::shared_ptr<SystemTable> MO_EXPORTS pythonSetup(const char* module_name);
        void MO_EXPORTS registerSetupFunction(std::function<void(void)>&& func);
        void MO_EXPORTS setLogLevel(const std::string& level);
        void MO_EXPORTS
        registerInterfaceSetupFunction(uint32_t interface_id,
                                       std::function<void(void)>&& interface_function,
                                       std::function<void(std::vector<IObjectConstructor*>&)>&& object_function);
        void MO_EXPORTS registerObjects();
        void MO_EXPORTS registerInterfaces();

        std::string MO_EXPORTS getModuleName();
        void MO_EXPORTS setModuleName(const std::string& name);

        template <class T>
        struct RegisterInterface
        {
            RegisterInterface(void (*setup)(), void (*construct)(std::vector<IObjectConstructor*>&))
            {
                registerInterfaceSetupFunction(T::getHash(), setup, construct);
            }
        };
    }
}
