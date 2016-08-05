#include "MetaObject/Parameters/UI/Qt/IHandler.hpp"
#include "MetaObject/Parameters/UI/Qt/SignalProxy.hpp"

using namespace mo::UI::qt;

IHandler::IHandler() : 
    paramMtx(nullptr), 
    proxy(new SignalProxy(this)), 
    _listener(nullptr) 
{
}

IHandler::~IHandler()
{
    if(proxy)
        delete proxy;
}
void IHandler::OnUiUpdate(QObject* sender) {}
void IHandler::OnUiUpdate(QObject* sender, double val) {}
void IHandler::OnUiUpdate(QObject* sender, int val) {}
void IHandler::OnUiUpdate(QObject* sender, bool val) {}
void IHandler::OnUiUpdate(QObject* sender, QString val) {}
void IHandler::OnUiUpdate(QObject* sender, int row, int col) {}
void IHandler::SetUpdateListener(UiUpdateListener* listener)
{
    _listener = listener;
}

std::function<void(void)>& IHandler::GetOnUpdate()
{

    return onUpdate;
}
std::vector<QWidget*> IHandler::GetUiWidgets(QWidget* parent)
{

    return std::vector<QWidget*>();
}
void IHandler::SetParamMtx(std::recursive_mutex* mtx)
{
    paramMtx = mtx;
}
std::recursive_mutex* IHandler::GetParamMtx()
{
    return paramMtx;
}

bool IHandler::UiUpdateRequired()
{
    return false;
}