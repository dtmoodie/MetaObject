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

https://github.com/dtmoodie/parameters
*/
#include "parameters/Parameter.hpp"


using namespace Parameters;

Parameter::Parameter(const std::string& name_, ParameterType type_):
    _name(name_), 
    flags(type_), 
    changed(false), 
    subscribers(0), 
    _current_time_index(-1),
    _ctx(nullptr)
{
    
}

Parameter::~Parameter()
{
    delete_signal(this);
}



Parameter* Parameter::SetName(const std::string& name_)
{
    _name = name_;
    return this;
}

Parameter* Parameter::SetTreeRoot(const std::string& treeRoot_)
{
    _tree_root = treeRoot_;
    return this;
}

Parameter* Parameter::SetTimeIndex(long long index)
{
    _current_time_index = index;
    return this;
}

Parameter* Parameter::SetContext(Signals::context* ctx)
{
    _ctx = ctx;
    return this;
}

const std::string& Parameter::GetName() const
{
    return _name;
}

const std::string& Parameter::GetTreeRoot() const
{
    return _tree_root;
}

const std::string Parameter::GetTreeName() const
{
    return _tree_root + ":" + _name;
}

long long Parameter::GetTimeIndex() const
{
    return _current_time_index;
}

Signals::context* Parameter::GetContext() const
{
    return _ctx;
}

std::shared_ptr<Signals::connection> Parameter::RegisterUpdateNotifier(std::function<void(Signals::context*, Parameter*)> f)
{
    return update_signal.connect(f);
}

std::shared_ptr<Signals::connection> Parameter::RegisterDeleteNotifier(std::function<void(Parameter*)> f)
{
    return delete_signal.connect(f);
}

bool Parameter::Update(Parameter* other, Signals::context* ctx)
{
    return false;
}

void Parameter::OnUpdate(Signals::context* ctx)
{
    changed = true;
    update_signal(ctx, this);
}

Parameter* Parameter::Commit(long long index_, Signals::context* ctx)
{
    _current_time_index = index_;
    changed = true;
    update_signal(ctx, this);
    return this;
}

std::recursive_mutex& Parameter::mtx()
{
    return _mtx;
}