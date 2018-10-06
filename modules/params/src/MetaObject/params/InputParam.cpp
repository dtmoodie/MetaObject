#include "MetaObject/params/InputParam.hpp"
#include <boost/thread/recursive_mutex.hpp>

using namespace mo;

InputParam::InputParam()
<<<<<<< HEAD
    : IParam("", mo::ParamFlags::Input_e)
=======
>>>>>>> 363c579de74f45297b4af110fb911020e1ab4d93
{
}

InputParam::~InputParam()
{
}

std::ostream& InputParam::print(std::ostream& os) const
{
    IParam* input = nullptr;
    {
        Lock lock(mtx());
        IParam::print(os);
        input = getInputParam();
    }

    if (input)
    {
        os << "\n";
        input->print(os);
    }
    return os;
}
