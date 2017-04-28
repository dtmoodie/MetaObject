#ifdef HAVE_QT5
#include "MetaObject/Params/UI/Qt/IHandler.hpp"
#include "MetaObject/Params/UI/Qt/SignalProxy.hpp"

using namespace mo::UI::qt;

IHandler::IHandler() : 
    paramMtx(nullptr), 
    proxy(new SignalProxy(this)), 
    _listener(nullptr) {
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
void IHandler::setUpdateListener(UiUpdateListener* listener){
    _listener = listener;
}

std::function<void(void)>& IHandler::getOnUpdate(){
    return onUpdate;
}
std::vector<QWidget*> IHandler::getUiWidgets(QWidget* parent){
    return std::vector<QWidget*>();
}
void IHandler::setParamMtx(mo::Mutex_t** mtx){
    paramMtx = mtx;
}
mo::Mutex_t* IHandler::getParamMtx(){
    if(paramMtx)
        return *paramMtx;
    return nullptr;
}

bool IHandler::uiUpdateRequired(){
    return false;
}
#endif