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
    class ICoordinateSystem;

    static const auto milli = boost::units::si::milli;
    static const auto nano = boost::units::si::nano;
    static const auto second = boost::units::si::second;
    static const auto millisecond = milli * second;
    static const auto nanosecond = nano * second;
    static const auto ms = millisecond;
    static const auto ns = nanosecond;
    typedef boost::units::quantity<boost::units::si::time> time_t;



    class Context;
	class Connection;
	class ISignal;
    class ISignalRelay;
	class TypeInfo;
    class IParameter;

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
    
    class UpdateToken;

    class MO_EXPORTS IParameter: boost::noncopyable
    {
    public:
        typedef std::shared_ptr<IParameter> Ptr;
        typedef TypedSlot<void(Context*, IParameter*)> update_f;
        typedef TypedSlot<void(IParameter const*)> delete_f;

        IParameter(const std::string& name_  = "",
                   ParameterType      flags_ = Control_e,
                   mo::time_t         ts_    = -1 * mo::second,
                   Context*           ctx_   = nullptr,
                   size_t             fn_    = std::numeric_limits<size_t>::max());

        virtual ~IParameter();

        IParameter*         SetName(const std::string& name_);
		IParameter*         SetTreeRoot(const std::string& tree_root_);
        IParameter*         SetContext(Context* ctx);
        IParameter*         SetFrameNumber(size_t fn);
        IParameter*         SetTimestamp(mo::time_t ts);
        IParameter*         SetCoordinateSystem(ICoordinateSystem* cs_);

        const std::string& GetName()      const;
        const std::string  GetTreeName()  const;
        const std::string& GetTreeRoot()  const;
        mo::time_t         GetTimestamp() const;
        size_t             GetFrameNumber() const;
        Context*           GetContext()   const;
        ICoordinateSystem* GetCoordinateSystem() const;

		void Subscribe();
		void Unsubscribe();
		bool HasSubscriptions() const;

        // Implemented in concrete type
        virtual const TypeInfo&     GetTypeInfo() const = 0;

        // Update with the values from another parameter
        virtual bool Update(IParameter* other);
        virtual Ptr  DeepCopy() const;

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

        IParameter* Commit(mo::time_t         ts_   = -1 * mo::second,
                           Context*           ctx_  = nullptr,
                           size_t             fn_   = std::numeric_limits<size_t>::max(),
                           ICoordinateSystem* cs_   = nullptr);
        
        template<class Archive> void serialize(Archive& ar);


        boost::recursive_mutex& mtx();
        void SetMtx(boost::recursive_mutex* mtx);

		void SetFlags(ParameterType flags_);
		void AppendFlags(ParameterType flags_);
        bool CheckFlags(ParameterType flag);

        bool modified = false;
	protected:
        template<class T> friend class UI::qt::ParameterProxy;

        TypedSignal<void(Context*, IParameter*)> _update_signal;
        TypedSignal<void(IParameter const*)>	 _delete_signal;
        std::string                              _name;
        std::string                              _tree_root;
        boost::recursive_mutex*                  _mtx;
        Context*                                 _ctx; // Context of object that owns this parameter
        int                                      _subscribers;
        ParameterType                            _flags;
        bool                                     _owns_mutex;
        ICoordinateSystem*                       _cs;
        mo::time_t                               _ts;
        size_t                                     _fn;
    };

    template<typename Archive>
    void IParameter::serialize(Archive& ar)
    {
        ar(_name);
        ar(_tree_root);
        ar(_ts);
        ar(_fn);
        ar(_flags);
    }

    class UpdateToken
    {
    public:
        UpdateToken(IParameter& param);
        ~UpdateToken();
        UpdateToken& operator()(mo::time_t&& ts);
        UpdateToken& operator()(size_t fn);
        UpdateToken& operator()(Context* ctx);
        UpdateToken& operator()(ICoordinateSystem* cs);
    private:
        size_t _fn;
        mo::time_t _ts;
        ICoordinateSystem* _cs;
        Context* _ctx;
        IParameter& _param;
    };
}
