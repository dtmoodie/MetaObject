#include "MetaObject/params/InputParamAny.hpp"
#include "IDataContainer.hpp"
#include <MetaObject/thread/fiber_include.hpp>
#include <boost/thread/locks.hpp>

using namespace mo;

InputParamAny::InputParamAny(const std::string& name)
{
    this->setName(name);
    this->appendFlags(mo::ParamFlags::kINPUT);
}

bool InputParamAny::acceptsInput(mo::IParam*) const
{
    return true;
}

bool InputParamAny::acceptsType(const mo::TypeInfo&) const
{
    return true;
}
