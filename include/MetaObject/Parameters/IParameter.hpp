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

    static const auto milli = boost::units::si::milli;
    static const auto nano = boost::units::si::nano;
    static const auto micro = boost::units::si::micro;
    static const auto second = boost::units::si::second;
    static const auto millisecond = milli * second;
    static const auto nanosecond = nano * second;
    static const auto microseconds = micro * second;
    static const auto ms = millisecond;
    static const auto ns = nanosecond;
    static const auto us = microseconds;
    typedef boost::units::quantity<boost::units::si::time> time_t;

    namespace UI
    {
        namespace qt
        {
            template<class T> class ParameterProxy;
        }
    }

    class Context;
    class Connection;
    class ISignal;
    class ISignalRelay;
    class TypeInfo;
    class IParameter;
    class ICoordinateSystem;
    class UpdateToken;
    class ParameterOwner;
    template<class T> class TypedSignal;
    template<class T> class TypedSlot;
    template<class T> class TypedSignalRelay;

    typedef TypedSlot<void(Context*, IParameter*)> ParameterUpdateSlot;
    typedef TypedSlot<void(IParameter const*)> ParameterDeleteSlot;
    typedef std::shared_ptr<ParameterUpdateSlot> ParameterUpdateSlotPtr;
    typedef std::shared_ptr<ParameterDeleteSlot> ParameterDeleteSlotPtr;

    MO_KEYWORD_INPUT(timestamp, mo::time_t)
    MO_KEYWORD_INPUT(frame_number, size_t)
    MO_KEYWORD_INPUT(coordinate_system, ICoordinateSystem*)
    MO_KEYWORD_INPUT(context, Context*)
    MO_KEYWORD_INPUT(parameter_name, std::string)
    MO_KEYWORD_INPUT(tree_root, std::string)
    MO_KEYWORD_INPUT(parameter_flags, ParameterType)

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
            if(const mo::time_t* ts = GetKeywordInputOptional<tag::timestamp>(args...))
                _ts = *ts;
            _fn        = GetKeywordInputDefault<tag::frame_number>(-1, args...);
            _ctx       = GetKeywordInputDefault<tag::context>(nullptr, args...);
            _cs        = GetKeywordInputDefault<tag::coordinate_system>(nullptr, args...);
            _flags     = GetKeywordInputDefault<tag::parameter_flags>(mo::Control_e, args...);
            _tree_root = GetKeywordInputDefault<tag::tree_root>("", args...);
        }

        IParameter(const std::string& name_        = "",
                   ParameterType      flags_       = Control_e,
                   boost::optional<mo::time_t> ts_ = boost::optional<mo::time_t>(),
                   Context*           ctx_         = nullptr,
                   size_t             fn_          = -1);

        virtual ~IParameter();

        // Get the name of this parameter
        IParameter*         SetName(const std::string& name_);
        // Get the root to the name of this parameter
        // IE: for objname:paramname, set objname
        IParameter*         SetTreeRoot(const std::string& tree_root_);
        // Set the compute context of this parameter
        IParameter*         SetContext(Context* ctx);
        // Set the frame number for this parameter
        IParameter*         SetFrameNumber(size_t fn);
        // Set the timestamp for this parameter
        IParameter*         SetTimestamp(mo::time_t ts);
        // Set the coordinate system for this parameter
        IParameter*         SetCoordinateSystem(ICoordinateSystem* cs_);
        // Get the name of this parameter
        const std::string&          GetName()      const;
        // Get the name of this parmaeter appended with the tree root
        // IE root_name:param_name
        const std::string           GetTreeName()  const;
        // Get the tree root of this parameter, ie the name of the owning parent object
        const std::string&          GetTreeRoot()  const;
        // Get the timestamp of this parameter, may not exist for all parameters
        boost::optional<mo::time_t> GetTimestamp() const;
        // Get the frame number for this parameter
        // This value is initialized to zero and incremented on every
        // update call unless specifically set in the update call
        size_t                      GetFrameNumber() const;
        // Get the compute context of this parameter
        Context*                    GetContext()   const;
        // Get the coordinate system of this parameter
        ICoordinateSystem*          GetCoordinateSystem() const;

        // Subscribe to this parameter as an output
        void Subscribe();
        // unsubscribe to this parameter as an output
        void Unsubscribe();
        // Determine if there are any input parameters using this parameter as an output
        bool HasSubscriptions() const;

        // Implemented in concrete type
        virtual const TypeInfo&     GetTypeInfo() const = 0;

        // these two functions are not used anywhere, and thus not tested
        // Update with the values from another parameter
        virtual bool Update(IParameter* other);
        // Create a deep copy of a parameter including copying of any underlying data
        virtual Ptr  DeepCopy() const;

        // Register slots to be called on update of this parameter
        std::shared_ptr<Connection> RegisterUpdateNotifier(update_f* f);
        std::shared_ptr<Connection> RegisterUpdateNotifier(ISlot* f);
        std::shared_ptr<Connection> RegisterUpdateNotifier(std::shared_ptr<ISignalRelay> relay);
        std::shared_ptr<Connection> RegisterUpdateNotifier(std::shared_ptr<TypedSignalRelay<void(Context*, IParameter*)>>& relay);

        // Register slots to be called on delete of this parameter
        std::shared_ptr<Connection> RegisterDeleteNotifier(delete_f* f);
        std::shared_ptr<Connection> RegisterDeleteNotifier(ISlot* f);
        std::shared_ptr<Connection> RegisterDeleteNotifier(std::shared_ptr<ISignalRelay> relay);
        std::shared_ptr<Connection> RegisterDeleteNotifier(std::shared_ptr<TypedSignalRelay<void(IParameter const*)>>& relay);


        // Sets changed to true and emits update signal
        // ctx is the context from which the update occured
        void OnUpdate(Context* ctx = nullptr);

        // Commit changes to a parameter, updates underlying meta info and emits signals accordingly
        IParameter* Commit(boost::optional<mo::time_t> ts_   = boost::optional<mo::time_t>(), // The timestamp of the new data
                           Context*                    ctx_  = Context::GetDefaultThreadContext(),                       // The context from which the data was updated
                           boost::optional<size_t>     fn_   = boost::optional<size_t>(),     // The frame number of the update
                           ICoordinateSystem*          cs_   = nullptr);                      // The coordinate system of the update

        // Used for cereal serialization
        template<class Archive> void serialize(Archive& ar);

        // Get reference to parameter mutex.  If SetMtx was called, this will point to the mutex that was set
        boost::recursive_mutex& mtx();
        // Use this to share a mutex with an owning object, ie a parent.
        void SetMtx(boost::recursive_mutex* mtx);

        // Set flags of the parameter
        void SetFlags(ParameterType flags_);
        // Append a flag to the parameter
        void AppendFlags(ParameterType flags_);
        // Check if a single flag is set
        bool CheckFlags(ParameterType flag) const;

        // Set to true if modified by the user interface etc, set to false by the owning object.
        bool									_modified = false;
    protected:
        friend class ParameterOwner;
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
namespace cereal
{
    template<class AR> void save(AR& ar, const mo::time_t& time)
    {
        ar(time.value());
    }
    template<class AR> void load(AR& ar, mo::time_t& time)
    {
        double value;
        ar(value);
        time.from_value(value);
    }
}
