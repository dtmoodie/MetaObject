#include "MetaObject/Parameters/InputParameter.hpp"
using namespace mo;

InputParameter::InputParameter():
    IParameter("", mo::Input_e)
{

}

InputParameter::~InputParameter()
{

}

bool InputParameter::GetInput(boost::optional<mo::time_t> ts, size_t* fn)
{
    return false;
}

bool InputParameter::GetInput(size_t fn, boost::optional<mo::time_t>* ts)
{
    return false;
}
