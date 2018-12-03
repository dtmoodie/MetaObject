#include "MetaObject/params/InputParamAny.hpp"
#include "IDataContainer.hpp"
#include <boost/thread/locks.hpp>
#include <boost/fiber/recursive_timed_mutex.hpp>
using namespace mo;

InputParamAny::InputParamAny(const std::string& name)
{
    this->setName(name);
    this->appendFlags(mo::ParamFlags::Input_e);
}

bool InputParamAny::acceptsInput(mo::IParam*) const
{
    return true;
}

bool InputParamAny::acceptsType(const mo::TypeInfo&) const
{
    return true;
}
