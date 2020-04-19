#include "MetaObject/params/SubscriberAny.hpp"
#include "IDataContainer.hpp"
#include <MetaObject/thread/fiber_include.hpp>
#include <boost/thread/locks.hpp>

namespace mo
{

    SubscriberAny::SubscriberAny(const std::string& name)
    {
        this->setName(name);
        this->appendFlags(ParamFlags::kINPUT);
    }

    bool SubscriberAny::acceptsPublisher(const IPublisher&) const
    {
        return true;
    }

    bool SubscriberAny::acceptsType(const TypeInfo&) const
    {
        return true;
    }

} // namespace mo