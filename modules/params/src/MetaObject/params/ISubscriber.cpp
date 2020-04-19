#include <MetaObject/params/ISubscriber.hpp>

namespace mo
{
    ISubscriber::ISubscriber()
    {
    }

    ISubscriber::~ISubscriber()
    {
    }

    OptionalTime ISubscriber::getNewestTimestamp() const
    {
        auto header = this->getNewestHeader();
        if (header)
        {
            return header->timestamp;
        }
        return {};
    }
} // namespace mo
