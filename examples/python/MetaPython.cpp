#include "boost/python.hpp"
#include "MetaObject/Python/Python.hpp"
#include "MetaObject/logging/logging.hpp"
#include <iostream>
#include "MetaObject/object/MetaObjectFactory.hpp"
#include <boost/python/to_python_converter.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"

namespace bp = boost::python;

struct example_python_object: public mo::IMetaObject
{
    MO_BEGIN(example_python_object);
        PARAM(int, test, 5);
    MO_END;
};

MO_REGISTER_OBJECT(example_python_object)
struct NullDeleter
{
    void operator()(const void*){}
};

boost::shared_ptr<mo::MetaObjectFactory> GetObjectFactory()
{
    return boost::shared_ptr<mo::MetaObjectFactory>(mo::MetaObjectFactory::instance(), NullDeleter());
}

template <typename T> T* get_pointer(rcc::shared_ptr<T>& p) {
    //notice the const_cast<> at this point
    //for some unknown reason, bp likes to have it like that
    return const_cast<T*>(p.get());
}

template<typename T> const T* get_pointer(const rcc::shared_ptr<T> & p)
{
    return const_cast<T*>(p.get());
}

namespace boost { 
    /*template <>
    example_python_object const volatile * get_pointer<class example_python_object const volatile >(
        class example_python_object const volatile *c)
    {
        return c;
    }*/
    namespace python {

    template <typename T> struct pointee<rcc::shared_ptr<T>> 
    {
        typedef T type;
    };


} }


BOOST_PYTHON_MODULE(MetaPython)
{
    mo::MetaObjectFactory::instance()->registerTranslationUnit();

    boost::python::scope().attr("__version__") = "0.1";

    mo::PythonClassRegistry::SetupPythonModule();

    bp::class_<mo::MetaObjectFactory, boost::shared_ptr<mo::MetaObjectFactory>, boost::noncopyable>("MetaObjectFactory", bp::no_init)
        .def("Instance", &GetObjectFactory).staticmethod("Instance")
        .def("ListConstructableObjects", &mo::MetaObjectFactory::listConstructableObjects)
        .def("LoadPlugin", &mo::MetaObjectFactory::loadPlugin);

    bp::class_<std::vector<std::string> >("StringVec")
        .def(bp::vector_indexing_suite<std::vector<std::string> >());

}