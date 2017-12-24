#include <boost/mpl/vector.hpp>
#include <functional>
namespace boost
{
  namespace python
  {
    namespace detail
    {

      template <class T, class... Args>
      inline boost::mpl::vector<T, Args...>
        get_signature(std::function<T(Args...)>, void* = 0)
      {
        return boost::mpl::vector<T, Args...>();
      }

    }
  }
}

#include "PythonSetup.hpp"
#include <boost/python.hpp>
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/params/InputParam.hpp"
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <RuntimeObjectSystem/IObjectInfo.h>
#include <boost/python/return_internal_reference.hpp>
#include <boost/functional.hpp>
#include <boost/python/signature.hpp>
#include "rcc_ptr.hpp"
#include "lambda.hpp"



namespace mo
{
    namespace python
    {
        void setupInterface()
        {

            boost::python::class_<IMetaObject, rcc::shared_ptr<IMetaObject>, boost::noncopyable> bpobj("IMetaObject", boost::python::no_init);
            bpobj.def("getParams",
                      static_cast<std::vector<IParam*>(IMetaObject::*)(const std::string&) const>(&IMetaObject::getParams),
                      (boost::python::arg("name_filter") = ""));
            bpobj.def("getInputs",
                      static_cast<std::vector<InputParam*>(IMetaObject::*)(const std::string&) const>(&IMetaObject::getInputs),
                      (boost::python::arg("name_filter") = ""));
            bpobj.def("getOutputs",
                      static_cast<ParamVec_t(IMetaObject::*)(const std::string&) const>(&IMetaObject::getOutputs),
                      (boost::python::arg("name_filter") = ""));

            bpobj.def("getParam",
                      static_cast<IParam*(IMetaObject::*)(const std::string&) const>(&IMetaObject::getParam),
                      (boost::python::arg("name")), boost::python::return_internal_reference<>());
            bpobj.def("getInput",
                      static_cast<InputParam*(IMetaObject::*)(const std::string&) const>(&IMetaObject::getInput),
                      (boost::python::arg("name")), boost::python::return_internal_reference<>());
            bpobj.def("getOutput",
                      static_cast<IParam*(IMetaObject::*)(const std::string&) const>(&IMetaObject::getOutput),
                      (boost::python::arg("name")), boost::python::return_internal_reference<>());

            bpobj.def("getContext", &IMetaObject::getContext);

        }

        rcc::shared_ptr<MetaObject> constructObject(IObjectConstructor* ctr)
        {
            rcc::shared_ptr<MetaObject> output;
            auto obj = ctr->Construct();
            if(obj)
            {
                output = obj;
                output->Init(true);
            }
            return output;
        }

        void setupObjects(std::vector<IObjectConstructor*>& ctrs)
        {
            boost::python::object module(boost::python::handle<>(boost::python::borrowed(PyImport_AddModule("metaobject.object"))));

            boost::python::import("metaobject").attr("object") = module;
            boost::python::scope plugins_scope = module;

            for(auto itr = ctrs.begin(); itr != ctrs.end();)
            {
                IObjectInfo* info = (*itr)->GetObjectInfo();
                if(info->InheritsFrom(IMetaObject::s_interfaceID))
                {
                    boost::python::class_<MetaObject, rcc::shared_ptr<MetaObject>, boost::python::bases<IMetaObject>, boost::noncopyable> bpobj(info->GetObjectName().c_str(),
                                                                                                  info->Print().c_str(),
                                                                                                  boost::python::no_init);
                    bpobj.def("__init__",
                              boost::python::make_constructor(std::function<rcc::shared_ptr<MetaObject>()>(std::bind(&constructObject, *itr))));
                    boost::python::import("metaobject").attr("object").attr(info->GetObjectName().c_str()) = bpobj;
                    itr = ctrs.erase(itr);
                }else
                {
                    ++itr;
                }
            }
        }

        static RegisterInterface<IMetaObject> g_register(&setupInterface, &setupObjects);
    }
}
