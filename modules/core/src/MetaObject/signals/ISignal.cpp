#include "MetaObject/signals/ISignal.hpp"
//#include "MetaObject/object/IMetaObject.hpp"
using namespace mo;

Context* ISignal::getContext() const{
    return _ctx;
    if(_parent){
        //return _parent->getContext();
    }
    return nullptr;
}
void ISignal::setContext(Context*  ctx){
    _ctx = ctx;
}

IMetaObject* ISignal::getParent() const{
    return _parent;
}

void ISignal::setParent(IMetaObject* parent){
    _parent = parent;
}
