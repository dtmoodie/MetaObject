#include "MetaObject/params/InputParam.hpp"
using namespace mo;

InputParam::InputParam() : IParam("", mo::ParamFlags::Input_e)
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

std::ostream& InputParam::print(std::ostream& os) const
{
    IParam::print(os);
    auto input = getInputParam();
    if (input)
    {
        os << "\n";
        input->print(os);
    }
    return os;
}
