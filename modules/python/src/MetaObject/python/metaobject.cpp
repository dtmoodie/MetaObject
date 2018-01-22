#include <boost/mpl/vector.hpp>
#include <functional>

namespace boost
{
    namespace python
    {
        namespace detail
        {

            template <class T, class... Args>
            inline boost::mpl::vector<T, Args...> get_signature(std::function<T(Args...)>, void* = 0)
            {
                return boost::mpl::vector<T, Args...>();
            }
        }
    }
}

#include "MetaObject.hpp"
#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/IMetaObjectInfo.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/params/ParamInfo.hpp"
#include "MetaObject/python/DataConverter.hpp"

#include "PythonSetup.hpp"
#include "lambda.hpp"
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
    namespace python
    {
        namespace detail
        {
			std::string getName(IObjectConstructor* ctr) { return ctr->GetName(); }
			std::string getCompiledPath(IObjectConstructor* ctr) { return ctr->GetCompiledPath(); }
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
        }
        void setupInterface()
        {

            boost::python::class_<IMetaObject, rcc::shared_ptr<IMetaObject>, boost::noncopyable> bpobj(
                "IMetaObject", boost::python::no_init);
            
            bpobj.def(
                "getParams",
                static_cast<std::vector<IParam*> (IMetaObject::*)(const std::string&) const>(&IMetaObject::getParams),
                (boost::python::arg("name_filter") = ""));
            
            bpobj.def("getInputs",
                      static_cast<std::vector<InputParam*> (IMetaObject::*)(const std::string&) const>(
                          &IMetaObject::getInputs),
                      (boost::python::arg("name_filter") = ""));
            
            bpobj.def("getOutputs",
                      static_cast<ParamVec_t (IMetaObject::*)(const std::string&) const>(&IMetaObject::getOutputs),
                      (boost::python::arg("name_filter") = ""));

            bpobj.def("getParam",
                      static_cast<IParam* (IMetaObject::*)(const std::string&)const>(&IMetaObject::getParam),
                      (boost::python::arg("name")),
                      boost::python::return_internal_reference<>());
            
            bpobj.def("getInput",
                      static_cast<InputParam* (IMetaObject::*)(const std::string&)const>(&IMetaObject::getInput),
                      (boost::python::arg("name")),
                      boost::python::return_internal_reference<>());

            bpobj.def("getOutput",
                      static_cast<IParam* (IMetaObject::*)(const std::string&)const>(&IMetaObject::getOutput),
                      (boost::python::arg("name")),
                      boost::python::return_internal_reference<>());

            bpobj.def("getContext", &IMetaObject::getContext);
            bpobj.def("setContext", &IMetaObject::setContext);


            boost::python::class_<IObjectConstructor, IObjectConstructor*, boost::noncopyable> ctrobj(
                "IObjectConstructor", boost::python::no_init);
        }

        rcc::shared_ptr<MetaObject> constructObject(IObjectConstructor* ctr)
        {
            rcc::shared_ptr<MetaObject> output;
            auto obj = ctr->Construct();
            if (obj)
            {
                output = obj;
                output->Init(true);
            }
            return output;
        }

        IObjectConstructor* getCtr(IObjectConstructor* ctr) { return ctr; }

        template<class T>
        bool setParamHelper(python::DataConverterRegistry::Set_t setter, std::string name,
                            T& obj, const boost::python::object& python_obj)
        {
            auto param = obj.getParamOptional(name);
            if (param)
            {
                return setter(param, python_obj);
            }
            return false;
        }

        template<class T>
        boost::python::object getParamHelper(python::DataConverterRegistry::Get_t getter, std::string name,
            const T& obj)
        {
            auto param = obj.getParamOptional(name);
            if (param)
            {
                return getter(param);
            }
            return {};
        }

        void setupObjects(std::vector<IObjectConstructor*>& ctrs)
        {
            boost::python::object module(
                boost::python::handle<>(boost::python::borrowed(PyImport_AddModule((mo::python::module_name + ".object").c_str()))));

            boost::python::import(mo::python::module_name.c_str()).attr("object") = module;
            boost::python::scope plugins_scope = module;

            for (auto itr = ctrs.begin(); itr != ctrs.end();)
            {
                IObjectInfo* info = (*itr)->GetObjectInfo();
                if (info->InheritsFrom(IMetaObject::s_interfaceID))
                {
                    boost::python::class_<MetaObject,
                                          rcc::shared_ptr<MetaObject>,
                                          boost::python::bases<IMetaObject>,
                                          boost::noncopyable>
                        bpobj(info->GetObjectName().c_str(), info->Print().c_str(), boost::python::no_init);
                    bpobj.def("__init__",
                              boost::python::make_constructor(
                                  std::function<rcc::shared_ptr<MetaObject>()>(std::bind(&constructObject, *itr))));
                    boost::python::object ctr = mo::makeConstructor<IMetaObject>(*itr);
                    if (ctr)
                    {
                        bpobj.def("__init__", ctr);
                    }
                    
                    auto minfo = dynamic_cast<IMetaObjectInfo*>(info);
                    if (minfo)
                    {
                        std::vector<ParamInfo*> param_infos = minfo->getParamInfo();
                        for (auto param_info : param_infos)
                        {
                            auto setter = python::DataConverterRegistry::instance()->getSetter(param_info->data_type);
                            auto getter = python::DataConverterRegistry::instance()->getGetter(param_info->data_type);
                            if (setter && getter)
                            {
                                bpobj.def(("get_" + param_info->name).c_str(), std::function<boost::python::object(const MetaObject&)>(std::bind(getParamHelper<MetaObject>, getter, param_info->name, std::placeholders::_1)));
                                bpobj.def(("set_" + param_info->name).c_str(), std::function<bool(MetaObject&, const boost::python::object&)>(std::bind(setParamHelper<MetaObject>, setter, param_info->name, std::placeholders::_1, std::placeholders::_2)));
                            }
                        }
                    }
                    
                    boost::python::import(mo::python::module_name.c_str()).attr("object").attr(info->GetObjectName().c_str()) = bpobj;
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
    }
}
