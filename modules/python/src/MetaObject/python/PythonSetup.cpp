#include "PythonSetup.hpp"
#include "DataConverter.hpp"
#include "MetaObject/params/ParamFactory.hpp"
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"

#include <boost/python.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/raw_function.hpp>

#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include <vector>

namespace boost
{
    inline const mo::ParamBase* get_pointer(const mo::ParamBase* ptr)
    {
        return ptr;
    }

    inline const mo::ICoordinateSystem* get_pointer(const mo::ICoordinateSystem* cs)
    {
        return cs;
    }

    inline const mo::ICoordinateSystem* get_pointer(const std::shared_ptr<mo::ICoordinateSystem>& ptr)
    {
        return ptr.get();
    }
}

namespace mo
{
    void setupEnums(const std::string& module_name);
    
    void setupDataTypes(const std::string& module_name);

    void setupPlugins(const std::string& module_name);

    inline const mo::ParamBase* get_pointer(const mo::ParamBase* ptr)
    {
        return ptr;
    }

    inline const mo::ICoordinateSystem* get_pointer(const mo::ICoordinateSystem* cs)
    {
        return cs;
    }

    inline mo::ICoordinateSystem* get_pointer(const std::shared_ptr<mo::ICoordinateSystem>& ptr)
    {
        return ptr.get();
    }

    static std::vector<std::function<void(void)>> setup_functions;
    void registerSetupFunction(std::function<void(void)>&& func)
    {
        setup_functions.emplace_back(std::move(func));
    }

    std::vector<std::string> listConstructableObjects()
    {
        auto ctrs = mo::MetaObjectFactory::instance()->getConstructors();
        std::vector<std::string> output;
        for(auto ctr : ctrs)
        {
            auto info = ctr->GetObjectInfo();
            if(info)
            {
                output.push_back(info->GetObjectName());
            }
        }
        return output;
    }

    std::vector<IObjectInfo*> listObjectInfos()
    {
        auto ctrs = mo::MetaObjectFactory::instance()->getConstructors();
        std::vector<IObjectInfo*> output;
        for (auto ctr : ctrs)
        {
            auto info = ctr->GetObjectInfo();
            if (info)
            {
                output.push_back(info);
            }
        }
        return output;
    }

    void pythonSetup(const char* module_name_)
    {
        std::string module_name(module_name_);
        setupEnums(module_name);
        setupDataTypes(module_name);
        setupPlugins(module_name);
        
        boost::python::def("listConstructableObjects", &listConstructableObjects);
        boost::python::def("listObjectInfos", &listObjectInfos);
    }
}


BOOST_PYTHON_MODULE(metaobject)
{
    mo::pythonSetup("metaobject");

}

