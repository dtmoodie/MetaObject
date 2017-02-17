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

https://github.com/dtmoodie/MetaObject
*/
#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/Enums.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include <boost/version.hpp>
#include <boost/units/systems/si.hpp>
#include <boost/units/systems/si/prefixes.hpp>
#include <boost/units/systems/si/time.hpp>
#include <boost/units/io.hpp>
#if BOOST_VERSION > 105400
#include <boost/core/noncopyable.hpp>
#else
#include <boost/noncopyable.hpp>
#endif
#include <string>
#include <memory>
namespace boost
{
    class recursive_mutex;
}

namespace mo
{
    class Context;
	class Connection;
	class ISignal;
    class ISignalRelay;
	class TypeInfo;
    class IParameter;
    class ICoordinateSystem;
    typedef boost::units::quantity<boost::units::si::time> time_t;
    static const auto milli = boost::units::si::milli;
    static const auto nano = boost::units::si::nano;
    static const auto second = boost::units::si::second;
    static const auto millisecond = milli * second;
    static const auto nanosecond = nano * second;
    static const auto ms = millisecond;
    static const auto ns = nanosecond;

    namespace UI
    {
        namespace qt
        {
            template<class T> class ParameterProxy;
        }
    }
    template<class T> class TypedSignal;
	template<class T> class TypedSlot;
	template<class T> class TypedSignalRelay;
    typedef TypedSlot<void(Context*, IParameter*)> ParameterUpdateSlot;
    typedef TypedSlot<void(IParameter const*)> ParameterDeleteSlot;
    typedef std::shared_ptr<ParameterUpdateSlot> ParameterUpdateSlotPtr;
    typedef std::shared_ptr<ParameterDeleteSlot> ParameterDeleteSlotPtr;
    

    class MO_EXPORTS IParameter: boost::noncopyable
    {
    public:
        typedef std::shared_ptr<IParameter> Ptr;
        typedef TypedSlot<void(Context*, IParameter*)> update_f;
        typedef TypedSlot<void(IParameter const*)> delete_f;

        IParameter(const std::string& name_ = "", 
                   ParameterType flags_ = Control_e, 
                   mo::time_t = -1 * mo::second, 
                   Context* ctx = nullptr, 
                   size_t frame_num_ = std::numeric_limits<size_t>::max());

        virtual ~IParameter();

        IParameter*         SetName(const std::string& name_);
		IParameter*         SetTreeRoot(const std::string& tree_root_);
        IParameter*         SetContext(Context* ctx);
        template<class T>
        IParameter* SetTimestamp(T&& ts)
        {
            _timestamp = time_t(ts);
            return this;
        }
        IParameter* SetFrameNumber(size_t fn);
        size_t GetFrameNumber() const;

        void SetCoordinateSystem(ICoordinateSystem* system);
        ICoordinateSystem* GetCoordinateSystem() const;


        const std::string& GetName()      const;
        const std::string  GetTreeName()  const;
        const std::string& GetTreeRoot()  const;
        time_t             GetTimestamp() const;
        Context*           GetContext()   const;

		void Subscribe();
		void Unsubscribe();
		bool HasSubscriptions() const;

        // Implemented in concrete type
        virtual const TypeInfo&     GetTypeInfo() const = 0;

        // Update with the values from another parameter
        virtual bool         Update(IParameter* other);
        virtual std::shared_ptr<IParameter> DeepCopy() const;

        std::shared_ptr<Connection> RegisterUpdateNotifier(update_f* f);
		std::shared_ptr<Connection> RegisterUpdateNotifier(ISlot* f);
		std::shared_ptr<Connection> RegisterUpdateNotifier(std::shared_ptr<ISignalRelay> relay);
		std::shared_ptr<Connection> RegisterUpdateNotifier(std::shared_ptr<TypedSignalRelay<void(Context*, IParameter*)>>& relay);

        std::shared_ptr<Connection> RegisterDeleteNotifier(delete_f* f);
		std::shared_ptr<Connection> RegisterDeleteNotifier(ISlot* f);
		std::shared_ptr<Connection> RegisterDeleteNotifier(std::shared_ptr<ISignalRelay> relay);
		std::shared_ptr<Connection> RegisterDeleteNotifier(std::shared_ptr<TypedSignalRelay<void(IParameter const*)>>& relay);


        // Sets changed to true and emits update signal
        void OnUpdate(Context* ctx = nullptr);
        IParameter* Commit(mo::time_t timestamp_= -1 * mo::second, Context* ctx = nullptr);
        
        template<class Archive> void serialize(Archive& ar);
        
        template<typename T> T*   GetDataPtr(mo::time_t timestamp_= -1 * mo::second, Context* ctx = nullptr);
        template<typename T> T    GetData(mo::time_t timestamp_= -1 * mo::second, Context* ctx = nullptr);
        template<typename T> bool GetData(T& value, mo::time_t timestamp_= -1 * mo::second, Context* ctx = nullptr);

        boost::recursive_mutex& mtx();
        void SetMtx(boost::recursive_mutex* mtx);

		void SetFlags(ParameterType flags_);
		void AppendFlags(ParameterType flags_);
		bool CheckFlags(ParameterType flag);

        bool modified;
		TypedSignal<void(Context*, IParameter*)> update_signal;
		TypedSignal<void(IParameter const*)>	 delete_signal;
	protected:
        template<class T> friend class UI::qt::ParameterProxy;
        std::string             _name;
        std::string             _tree_root;
        time_t                  _timestamp;
        size_t                  _sequence_number;
        boost::recursive_mutex* _mtx; 
        Context*                _ctx; // Context of object that owns this parameter
        int                     _subscribers = 0;
        ParameterType           _flags;
        bool                    _owns_mutex;
        ICoordinateSystem*      _cs;
    };

    template<typename Archive> void IParameter::serialize(Archive& ar)
    {
        ar(_name);
        ar(_tree_root);
        ar(_timestamp);
        ar(_flags);
    }

    template<typename T> class ITypedParameter;
    
    template<typename T> T* IParameter::GetDataPtr(mo::time_t ts_, Context* ctx_)
    {
		if (auto typed = dynamic_cast<ITypedParameter<T>*>(this))
            return typed->GetDataPtr(ts_, ctx_);
		return nullptr;
    }

    template<typename T> T IParameter::GetData(mo::time_t ts_, Context* ctx_)
	{
		if (auto typed = dynamic_cast<ITypedParameter<T>*>(this))
            return typed->GetData(ts_, ctx_);
#ifndef __CUDACC__
        //throw "Bad cast. Requested " << typeid(T).name() << " actual " << GetTypeInfo().name();
#else
        throw "Bad cast";
#endif
	}

    template<typename T> bool IParameter::GetData(T& value, mo::time_t ts_, Context* ctx_)
	{
		if (auto typed = dynamic_cast<ITypedParameter<T>*>(this))
            return typed->GetData(value, ts_, ctx_);
		return false;
	}
}
