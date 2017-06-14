#include "MetaObject/params/InputParam.hpp"
using namespace mo;

InputParam::InputParam():
    IParam("", mo::Input_e)
{

}

InputParam::~InputParam()
{

}

bool InputParam::getInput(const OptionalTime_t& ts, size_t* fn)
{
    return false;
}

bool InputParam::getInput(size_t fn, OptionalTime_t* ts)
{
    return false;
}
