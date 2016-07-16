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
#include "MetaObject/Parameters/IParameter.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include <algorithm>

using namespace mo;

IParameter::IParameter(const std::string& name_, ParameterType flags_, long long ts, Context* ctx) :
    _name(name_), 
    _flags(flags_), 
    modified(false), 
    _subscribers(0), 
    _timestamp(ts),
    _ctx(ctx)
{
    
}

IParameter::~IParameter()
{
	if(auto sig = delete_signal.lock())
        (*sig)(this);
}



IParameter* IParameter::SetName(const std::string& name_)
{
    _name = name_;
    return this;
}

IParameter* IParameter::SetTreeRoot(const std::string& treeRoot_)
{
    _tree_root = treeRoot_;
    return this;
}

IParameter* IParameter::SetTimestamp(long long ts)
{
    _timestamp = ts;
    return this;
}

IParameter* IParameter::SetContext(Context* ctx)
{
    _ctx = ctx;
    return this;
}

const std::string& IParameter::GetName() const
{
    return _name;
}

const std::string& IParameter::GetTreeRoot() const
{
    return _tree_root;
}

const std::string IParameter::GetTreeName() const
{
    return _tree_root + ":" + _name;
}

long long IParameter::GetTimestamp() const
{
    return _timestamp;
}

Context* IParameter::GetContext() const
{
    return _ctx;
}

std::shared_ptr<Connection> IParameter::RegisterUpdateNotifier(update_f f)
{
	std::shared_ptr<TypedSignal<void(Context*, IParameter*)>> sig(update_signal);
	if (sig)
	{

	}
	return std::shared_ptr<Connection>();
}

std::shared_ptr<Connection> IParameter::RegisterDeleteNotifier(delete_f f)
{
	std::shared_ptr<TypedSignal<void(IParameter*)>> sig(delete_signal);
	if (sig)
	{

	}
	return std::shared_ptr<Connection>();
}

bool IParameter::Update(IParameter* other)
{
    return false;
}

void IParameter::OnUpdate(Context* ctx)
{
    modified = true;
	if (!update_signal.expired())
	{
		std::shared_ptr<TypedSignal<void(Context*, IParameter*)>> sig(update_signal);
		(*sig)(ctx, this);
	}
}

IParameter* IParameter::Commit(long long index_, Context* ctx)
{
    _timestamp= index_;
    modified = true;
	if (!update_signal.expired())
	{
		std::shared_ptr<TypedSignal<void(Context*, IParameter*)>> sig(update_signal);
		//(*sig)(ctx, this);
	}
    return this;
}

std::recursive_mutex& IParameter::mtx()
{
    return _mtx;
}
void IParameter::Subscribe()
{
	--_subscribers;
	_subscribers = std::max(0, _subscribers);
}

void IParameter::Unsubscribe()
{
	++_subscribers;
}

bool IParameter::HasSubscriptions() const
{
	return _subscribers != 0;
}
void IParameter::SetFlags(ParameterType flags_)
{
	_flags = flags_;
}

void IParameter::AppendFlags(ParameterType flags_)
{
	_flags = ParameterType(_flags | flags_);
}

bool IParameter::CheckFlags(ParameterType flag)
{
	return _flags & flag;
}
std::shared_ptr<IParameter> IParameter::DeepCopy() const
{
    return std::shared_ptr<IParameter>();
}