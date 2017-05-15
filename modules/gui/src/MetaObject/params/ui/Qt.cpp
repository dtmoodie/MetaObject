/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/Params
*/

#include "MetaObject/params/UI/WidgetFactory.hpp"
#include "MetaObject/params/UI/Qt/DefaultProxy.hpp"
#include "MetaObject/logging/Log.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/params/IParam.hpp"

#include <map>

using namespace mo;
using namespace mo::UI;
using namespace mo::UI::qt;

struct WidgetFactory::impl
{
    std::map<TypeInfo, WidgetFactory::HandlerConstructor_f> registry;
};

WidgetFactory::WidgetFactory()
{
    _pimpl = new impl();
}
WidgetFactory* WidgetFactory::Instance()
{
    static WidgetFactory inst;
    return &inst;
}

void WidgetFactory::RegisterConstructor(const TypeInfo& type, WidgetFactory::HandlerConstructor_f f)
{
    LOG(trace) << "Registering type " << type.name();
    auto itr = _pimpl->registry.find(type);
    if(itr == _pimpl->registry.end())
    {
        _pimpl->registry[type] = f;
    }   
}
std::string print_types(std::map<TypeInfo, WidgetFactory::HandlerConstructor_f>& registry)
{
    std::stringstream ss;
    for(auto& item : registry)
    {
        ss << item.first.name() << ", ";
    }
    return ss.str();
}

std::shared_ptr<IParamProxy> WidgetFactory::CreateProxy(IParam* param)
{
    std::string typeName = param->getTypeInfo().name();
    std::string treeName = param->getTreeName();
    auto itr = _pimpl->registry.find(param->getTypeInfo());
    if (itr == _pimpl->registry.end())
    {
        LOG(debug) << "No Widget Factory registered for type " << typeName 
            << " unable to make widget for Param: " << treeName 
            << ".  Known types: " << print_types(_pimpl->registry);
            
        return std::shared_ptr<IParamProxy>(new DefaultProxy(param));
    }
    LOG(trace) << "Creating handler for " << typeName << " " << treeName;
    return itr->second(param);
}

