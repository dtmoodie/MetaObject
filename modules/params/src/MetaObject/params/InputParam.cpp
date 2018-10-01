#include "MetaObject/params/InputParam.hpp"
#include <boost/thread/recursive_mutex.hpp>

using namespace mo;

InputParam::InputParam()
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
