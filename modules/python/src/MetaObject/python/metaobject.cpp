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

        IObjectConstructor* getCtr(IObjectConstructor* ctr) { return ctr; }


        void setupObjects(std::vector<IObjectConstructor*>& ctrs)
        {
            auto module_name = mo::python::getModuleName();
            boost::python::object module(
                boost::python::handle<>(boost::python::borrowed(PyImport_AddModule((module_name + ".object").c_str()))));

            boost::python::import(module_name.c_str()).attr("object") = module;
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
                                  std::function<rcc::shared_ptr<MetaObject>()>(std::bind(&constructObject<MetaObject>, *itr))));
                    boost::python::object ctr = mo::makeConstructor<IMetaObject>(*itr);
                    if (ctr)
                    {
                        bpobj.def("__init__", ctr);
                    }
                    
                    auto minfo = dynamic_cast<IMetaObjectInfo*>(info);
                    if (minfo)
                    {
                        addParamAccessors<MetaObject>(bpobj, minfo);
                    }
                    
                    boost::python::import(module_name.c_str()).attr("object").attr(info->GetObjectName().c_str()) = bpobj;
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

        //static RegisterInterface<IMetaObject> g_register(&setupInterface, &setupObjects);
    }
}
