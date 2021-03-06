#include "MetaObject/params/ui/Qt/SignalProxy.hpp"
#include "MetaObject/params/ui/Qt/IHandler.hpp"

using namespace mo::UI::qt;

SignalProxy::SignalProxy(IHandler* handler_)
{
    handler = handler_;
    lastCallTime.start();
}

void SignalProxy::on_update()
{
    if (lastCallTime.elapsed() > 15)
    {
        lastCallTime.start();
        handler->onUiUpdate(sender());
    }
}

void SignalProxy::on_update(int val)
{
    if (lastCallTime.elapsed() > 15)
    {
        lastCallTime.start();
        handler->onUiUpdate(sender(), val);
    }
}

void SignalProxy::on_update(double val)
{
    if (lastCallTime.elapsed() > 15)
    {
        lastCallTime.start();
        handler->onUiUpdate(sender(), val);
    }
}

void SignalProxy::on_update(bool val)
{
    if (lastCallTime.elapsed() > 15)
    {
        lastCallTime.start();
        handler->onUiUpdate(sender(), val);
    }
}

void SignalProxy::on_update(QString val)
{
    if (lastCallTime.elapsed() > 15)
    {
        lastCallTime.start();
        handler->onUiUpdate(sender(), val);
    }
}

void SignalProxy::on_update(int row, int col)
{
    if (lastCallTime.elapsed() > 15)
    {
        lastCallTime.start();
        handler->onUiUpdate(sender(), row, col);
    }
}
