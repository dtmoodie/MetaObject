#ifdef HAVE_QT5
#include "MetaObject/Params/UI/Qt/IHandler.hpp"
#include "MetaObject/Params/UI/Qt/SignalProxy.hpp"

using namespace mo::UI::qt;

IHandler::IHandler() : 
    proxy(new SignalProxy(this)) {
}

IHandler::~IHandler(){
    if(proxy)
        delete proxy;
}
void IHandler::onUiUpdate(QObject* sender) {}
void IHandler::onUiUpdate(QObject* sender, double val) {}
void IHandler::onUiUpdate(QObject* sender, int val) {}
void IHandler::onUiUpdate(QObject* sender, bool val) {}
void IHandler::onUiUpdate(QObject* sender, QString val) {}
void IHandler::onUiUpdate(QObject* sender, int row, int col) {}

std::function<void(void)>& IHandler::getOnUpdate(){
    return onUpdate;
}
std::vector<QWidget*> IHandler::getUiWidgets(QWidget* parent){
    return std::vector<QWidget*>();
}

#endif