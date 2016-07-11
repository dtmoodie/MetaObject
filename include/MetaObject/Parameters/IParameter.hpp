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
#pragma once

#include "Parameter_def.hpp"
#include <string>
#include <parameters/LokiTypeInfo.h>
#include <mutex>
#include <signals/signal.h>
namespace cv
{
    namespace cuda
    {
        class Stream;
    }
}
namespace Signals
{
    class context;
    class connection;
}
namespace Parameters
{
    enum ParameterType
    {
        kNone = 0,
        kInput = 1,
        kOutput = 2,
        kState = 4,
        kControl = 8,
        kBuffer = 16
    };
    class PARAMETER_EXPORTS Parameter
    {
    public:
        typedef std::shared_ptr<Parameter> Ptr;
        typedef std::function<void(Signals::context*, Parameter*)> update_f;
        typedef std::function<void(Parameter*)> delete_f;
        Parameter(const std::string& name_ = "", ParameterType flags_ = kControl);
        virtual ~Parameter();
        
        Parameter*         SetName(const std::string& name_);
        Parameter*         SetTreeRoot(const std::string& tree_root_);
        Parameter*         SetContext(Signals::context* ctx);
        virtual Parameter* SetTimeIndex(long long index_ = -1);

        const std::string& GetName() const;
        const std::string  GetTreeName() const;
        const std::string& GetTreeRoot() const;
        long long          GetTimeIndex() const;
        Signals::context*  GetContext() const;

        // Implemented in concrete type
        virtual Loki::TypeInfo     GetTypeInfo() const = 0;

        // Update with the values from another parameter
        virtual bool        Update(Parameter* other, Signals::context* ctx = nullptr);
        virtual Parameter*  DeepCopy() const = 0;

        std::shared_ptr<Signals::connection> RegisterUpdateNotifier(update_f f);
        std::shared_ptr<Signals::connection> RegisterDeleteNotifier(delete_f f);

        // Sets changed to true and emits update signal
        void OnUpdate(Signals::context* ctx = nullptr);
        Parameter* Commit(long long index_ = -1, Signals::context* ctx = nullptr);
        
        template<class Archive> void serialize(Archive& ar);
        template<typename T> T* GetData(long long time_index_ = -1);

        virtual std::recursive_mutex& mtx();

        ParameterType flags;
        bool changed;
        unsigned short subscribers;
        Signals::typed_signal_base<void(Signals::context*, Parameter*)> update_signal;
        Signals::typed_signal_base<void(Parameter*)> delete_signal;
    protected:
        std::string          _name;
        std::string          _tree_root;
        long long            _current_time_index;
        std::recursive_mutex _mtx;
        Signals::context*    _ctx; // Context of object that owns this parameter
    };

    struct ParameterInfo
    {
        Loki::TypeInfo data_type;
        std::string name;
        std::string tooltip;
        std::string description;
        ParameterType type_flags;
    };

    template<typename Archive> void Parameter::serialize(Archive& ar)
    {
        ar(_name);
        ar(_tree_root);
        ar(_current_time_index);
        ar(flags);
    }
    template<typename T> class ITypedParameter;
    template<typename T> T* Parameter::GetData(long long time_index_)
    {
        return static_cast<ITypedParameter<T>*>(this)->GetData(time_index_);
    }
}