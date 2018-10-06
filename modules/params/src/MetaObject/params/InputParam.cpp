#include "MetaObject/params/InputParam.hpp"
#include <boost/thread/recursive_mutex.hpp>

using namespace mo;

InputParam::InputParam()
    : IParam("", mo::ParamFlags::Input_e)
{
}

InputParam::~InputParam()
{
}

std::ostream& InputParam::print(std::ostream& os) const
{
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        IParam::print(os);
    }

    auto input = getInputParam();
    if (input)
    {
        os << "\n";
        input->print(os);
    }
    return os;
}
