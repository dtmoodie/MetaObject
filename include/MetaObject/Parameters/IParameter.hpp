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
#include "NamedParameter.hpp"

#include <boost/version.hpp>
#include <boost/units/systems/si.hpp>
#include <boost/units/systems/si/prefixes.hpp>
#include <boost/units/systems/si/time.hpp>
#include <boost/units/io.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>


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

    /*BOOST_PARAMETER_NAME(timestamp)
    BOOST_PARAMETER_NAME(frame_number)
    BOOST_PARAMETER_NAME(coordinate_system)
    BOOST_PARAMETER_NAME(context)
    BOOST_PARAMETER_NAME(parameter_name)
    BOOST_PARAMETER_NAME(parameter_flags)*/

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

    MO_KEYWORD_INPUT(timestamp, boost::optional<mo::time_t>)
    MO_KEYWORD_INPUT(frame_number,size_t)
    MO_KEYWORD_INPUT(coordinate_system, ICoordinateSystem*)
    MO_KEYWORD_INPUT(context, Context*)
    MO_KEYWORD_INPUT(parameter_name, std::string)
    MO_KEYWORD_INPUT(tree_root, std::string)
    MO_KEYWORD_INPUT(parameter_flags, ParameterType)

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

        template<class...Args>
        IParameter(const Args&... args)
        {
            _name      = GetKeywordInputDefault<tag::parameter_name>("unnamed", args...);
            _ts        = GetKeywordInputDefault<tag::timestamp>(boost::optional<mo::time_t>(), args...);
            _fn        = GetKeywordInputDefault<tag::frame_number>(0, args...);
            _ctx       = GetKeywordInputDefault<tag::context>(nullptr, args...);
            _cs        = GetKeywordInputDefault<tag::coordinate_system>(nullptr, args...);
            _flags     = GetKeywordInputDefault<tag::parameter_flags>(mo::Control_e, args...);
            _tree_root = GetKeywordInputDefault<tag::tree_root>("", args...);
        }

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

        const std::string&          GetName()      const;
        const std::string           GetTreeName()  const;
        const std::string&          GetTreeRoot()  const;
        boost::optional<mo::time_t> GetTimestamp() const;
        size_t                      GetFrameNumber() const;
        Context*                    GetContext()   const;
        ICoordinateSystem*          GetCoordinateSystem() const;

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

        IParameter* Commit(boost::optional<mo::time_t> ts_   = boost::optional<mo::time_t>(),
                           Context*                    ctx_  = nullptr,
                           boost::optional<size_t>     fn_   = boost::optional<size_t>(),
                           ICoordinateSystem*          cs_   = nullptr);
        
        template<class Archive> void serialize(Archive& ar);


        boost::recursive_mutex& mtx();
        void SetMtx(boost::recursive_mutex* mtx);

		void SetFlags(ParameterType flags_);
		void AppendFlags(ParameterType flags_);
        bool CheckFlags(ParameterType flag);

        bool modified = false;
	protected:
        template<class T> friend class UI::qt::ParameterProxy;
        boost::optional<mo::time_t>              _ts;
        size_t                                   _fn;
        ICoordinateSystem*                       _cs;
        Context*                                 _ctx; // Context of object that owns this parameter
        std::string                              _name;
        std::string                              _tree_root;
        ParameterType                            _flags;
        TypedSignal<void(Context*, IParameter*)> _update_signal;
        TypedSignal<void(IParameter const*)>	 _delete_signal;
        boost::recursive_mutex*                  _mtx = nullptr;
        int                                      _subscribers = 0;
        bool                                     _owns_mutex = false;
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
