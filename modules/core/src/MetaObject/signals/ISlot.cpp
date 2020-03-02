#include "MetaObject/signals/ISlot.hpp"
#include "MetaObject/signals/ISignalRelay.hpp"

using namespace mo;

ISlot::~ISlot()
{
    // TODO stream remove
}

IAsyncStream* ISlot::getStream() const
{
    return m_stream;
}
void ISlot::setStream(IAsyncStream& stream)
{
    m_stream = &stream;
}
