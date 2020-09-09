#include "lambda.hpp"

#include "MetaObject.hpp"
#include "MetaObject/object/IMetaObjectInfo.hpp"
#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/params/ISubscriber.hpp"
#include "MetaObject/params/ParamInfo.hpp"
#include "MetaObject/python/DataConverter.hpp"

#include "PythonSetup.hpp"

#include "rcc_ptr.hpp"
#include <RuntimeObjectSystem/IObjectInfo.h>
#include <RuntimeObjectSystem/RuntimeObjectSystem.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <boost/functional.hpp>
#include <boost/python.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/signature.hpp>

namespace mo
{

    boost::python::object getParam(const mo::IControlParam* param)
    {
        python::ControlParamGetter getter;
        param->save(getter);
        return getter.getObject();
    }

    bool setParam(mo::IControlParam* param, const boost::python::object& python_obj)
    {
        python::ControlParamSetter setter(python_obj);
        param->load(setter);
        return setter.success();
    }
    
    namespace python
    {
        namespace detail
        {
            std::string getName(IObjectConstructor* ctr)
            {
                return ctr->GetName();
            }
            std::string getCompiledPath(IObjectConstructor* ctr)
            {
                return ctr->GetCompiledPath();
            }
            std::vector<std::string> getInludeFiles(const IObjectConstructor* ctr)
            {
                std::vector<std::string> output;
                const size_t cnt = ctr->GetMaxNumIncludeFiles();
                for (size_t i = 0; i < cnt; ++i)
                {
                    output.emplace_back(ctr->GetIncludeFile(i));
                }
                return output;
            }
            std::vector<std::string> getLinkLibs(const IObjectConstructor* ctr)
            {
                std::vector<std::string> output;
                const size_t cnt = ctr->GetMaxNumLinkLibraries();
                for (size_t i = 0; i < cnt; ++i)
                {
                    output.emplace_back(ctr->GetLinkLibrary(i));
                }
                return output;
            }
            /*std::vector<std::string> getSourceDependencies(const IObjectConstructor* ctr)
            {
                    std::vector<std::string> output;
                    const size_t cnt = ctr->GetMaxNumSourceDependencies();
                    for (size_t i = 0; i < cnt; ++i)
                    {
                            output.emplace_back(ctr->GetSourceDependency(i));
                    }
                    return output;
            }*/
        } // namespace detail
        void setupInterface()
        {
            static bool setup = false;
            if (setup)
            {
                return;
            }
            MO_LOG(info, "Registering IMetaObject to python");
            boost::python::class_<IMetaObject, rcc::shared_ptr<IMetaObject>, boost::noncopyable> bpobj(
                "IMetaObject", boost::python::no_init);
            bpobj.def("__repr__", &printObject<IMetaObject>);
            bpobj.def("getParams",
                      static_cast<ParamVec_t (IMetaObject::*)(const std::string&)>(&IMetaObject::getParams),
                      (boost::python::arg("name_filter") = ""));

            bpobj.def("getInputs",
                      static_cast<std::vector<ISubscriber*> (IMetaObject::*)(const std::string&) const>(
                          &IMetaObject::getInputs),
                      (boost::python::arg("name_filter") = ""));

            bpobj.def(
                "getOutputs",
                static_cast<IMetaObject::PublisherVec_t (IMetaObject::*)(const std::string&)>(&IMetaObject::getOutputs),
                (boost::python::arg("name_filter") = ""));

            bpobj.def("getParam",
                      static_cast<IControlParam* (IMetaObject::*)(const std::string&)>(&IMetaObject::getParam),
                      (boost::python::arg("name")),
                      boost::python::return_internal_reference<>());

            bpobj.def("getInput",
                      static_cast<ISubscriber* (IMetaObject::*)(const std::string&)>(&IMetaObject::getInput),
                      (boost::python::arg("name")),
                      boost::python::return_internal_reference<>());

            bpobj.def("getOutput",
                      static_cast<IPublisher* (IMetaObject::*)(const std::string&)>(&IMetaObject::getOutput),
                      (boost::python::arg("name")),
                      boost::python::return_internal_reference<>());

            bpobj.def("getStream", &IMetaObject::getStream);
            bpobj.def("setStream", &IMetaObject::setStream);

            boost::python::class_<IObjectConstructor, IObjectConstructor*, boost::noncopyable> ctrobj(
                "IObjectConstructor", boost::python::no_init);
            setup = true;
        }

        IObjectConstructor* getCtr(IObjectConstructor* ctr)
        {
            return ctr;
        }

        void setupObjects(std::vector<IObjectConstructor*>& ctrs)
        {
            auto module_name = mo::python::getModuleName();
            boost::python::object module(boost::python::handle<>(
                boost::python::borrowed(PyImport_AddModule((module_name + ".object").c_str()))));

            boost::python::import(module_name.c_str()).attr("object") = module;
            boost::python::scope plugins_scope = module;

            for (auto itr = ctrs.begin(); itr != ctrs.end();)
            {
                const IObjectInfo* info = (*itr)->GetObjectInfo();
                if (info->InheritsFrom(IMetaObject::getHash()))
                {
                    const auto name = info->GetObjectName();
                    MO_LOG(debug, "Registering {} to python", name);
                    auto docstring = info->Print();
                    boost::python::class_<MetaObject,
                                          rcc::shared_ptr<MetaObject>,
                                          boost::python::bases<IMetaObject>,
                                          boost::noncopyable>
                        bpobj(name.c_str(), docstring.c_str(), boost::python::no_init);
                    bpobj.def("__init__",
                              boost::python::make_constructor(std::function<rcc::shared_ptr<MetaObject>()>(
                                  std::bind(&constructObject<MetaObject>, *itr))));
                    boost::python::object ctr = mo::makeConstructor<IMetaObject>(*itr);
                    if (ctr)
                    {
                        bpobj.def("__init__", ctr);
                    }

                    auto minfo = dynamic_cast<const IMetaObjectInfo*>(info);
                    if (minfo)
                    {
                        addParamAccessors<MetaObject>(bpobj, minfo);
                    }

                    boost::python::import(module_name.c_str()).attr("object").attr(info->GetObjectName().c_str()) =
                        bpobj;
                    itr = ctrs.erase(itr);
                }
                else
                {
                    ++itr;
                }
            }
            boost::python::implicitly_convertible<MetaObject*, IMetaObject*>();
            boost::python::implicitly_convertible<rcc::shared_ptr<MetaObject>, rcc::shared_ptr<IMetaObject>>();
        }

        static RegisterInterface<IMetaObject> g_register(&setupInterface, &setupObjects);
    } // namespace python
} // namespace mo
