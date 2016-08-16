#include "boost/python.hpp"
#include "MetaObject/Python/Python.hpp"
#include "MetaObject/Logging/Log.hpp"
#include <iostream>
BOOST_PYTHON_MODULE(MetaPython)
{
    LOG(debug) << "Setting up python stuffs";
    std::cout << "test";
    boost::python::scope().attr("__version__") = "0.1";
    mo::PythonClassRegistry::SetupPythonModule();
    
}