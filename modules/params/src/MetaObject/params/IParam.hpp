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
#include "MetaObject/core.hpp"
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/params/NamedParam.hpp"
#include "MetaObject/signals/TSignal.hpp"

#if BOOST_VERSION > 105400
#include <boost/core/noncopyable.hpp>
#else
#include <boost/noncopyable.hpp>
#endif

#include <memory>
#include <string>

namespace mo
{
    namespace UI
    {
        namespace qt
        {
            template <class T>
            class ParamProxy;
        } // namespace UI::qt
    }     // namespace UI

    MO_KEYWORD_INPUT(timestamp, mo::Time_t)
    MO_KEYWORD_INPUT(frame_number, size_t)
    MO_KEYWORD_INPUT(coordinate_system, const std::shared_ptr<ICoordinateSystem>)
    MO_KEYWORD_INPUT(context, Context*)
    MO_KEYWORD_INPUT(param_name, std::string)
    MO_KEYWORD_INPUT(tree_root, std::string)
    MO_KEYWORD_INPUT(param_flags, EnumClassBitset<ParamFlags>)

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#if defined(metaobject_params_EXPORTS)
#define PARAM_EXPORTS __declspec(dllexport)
#else
#define PARAM_EXPORTS __declspec(dllimport)
#endif
#elif defined __GNUC__ && __GNUC__ >= 4
#define PARAM_EXPORTS __attribute__((visibility("default")))
#else
#define PARAM_EXPORTS
#endif

    namespace tag
    {
        struct param;
    }
    namespace kwargs
    {
        template <>
        struct TaggedArgument<tag::param> : public TaggedBase
        {
            typedef tag::param TagType;
            explicit TaggedArgument(const IParam& val) : arg(&val) {}

            const void* get() const { return arg; }

          protected:
            const void* arg;
        };

        template <>
        struct PARAM_EXPORTS TKeyword<tag::param>
        {
            static TKeyword instance;
            TaggedArgument<tag::param> operator=(const IParam& data);
        };

    } // namespace kwargs

    namespace tag
    {
        struct param
        {
            typedef IParam Type;
            typedef const Type& ConstRef;
            typedef Type& Ref;
            typedef ConstRef StorageType;
            typedef const void* VoidType;
            template <typename T>
            static constexpr bool AllowedType()
            {
                return std::is_same<Type, T>::value;
            }
            static VoidType GetPtr(const Type& arg) { return &arg; }
            template <class T>
            static VoidType GetPtr(const T& arg)
            {
                (void)arg;
                return nullptr;
            }
        };
        static mo::kwargs::TKeyword<param>& _param = mo::kwargs::TKeyword<param>::instance;
    }

    typedef void(UpdateSig_t)(IParam*,
                              Context*,
                              OptionalTime_t,
                              size_t,
                              const std::shared_ptr<ICoordinateSystem>&,
                              UpdateFlags); // Sig for signature not signal
    typedef TSlot<UpdateSig_t> UpdateSlot_t;
    typedef TSignal<UpdateSig_t> UpdateSignal_t;
    typedef std::shared_ptr<UpdateSlot_t> UpdateSlotPtr_t;

    typedef TSlot<void(const IParam*)> DeleteSlot_t;
    typedef TSignal<void(const IParam*)> DeleteSignal_t;
    typedef std::shared_ptr<DeleteSlot_t> DeleteSlotPtr_t;

    class MO_EXPORTS ParamBase : public boost::noncopyable
    {
      public:
        virtual ~ParamBase();
        typedef std::shared_ptr<ParamBase> Ptr;
        typedef std::shared_ptr<const ParamBase> ConstPtr;
        virtual const std::string& getName() const = 0; // Get the name of this Param
        virtual const std::string
        getTreeName() const = 0; // Get the name of this parmaeter appended with the tree root. IE root_name:param_name
        virtual const std::string&
        getTreeRoot() const = 0; // Get the tree root of this Param, ie the name of the owning parent object
        virtual OptionalTime_t
        getTimestamp() const = 0;                  // Get the timestamp of this Param, may not exist for all Params
        virtual size_t getFrameNumber() const = 0; // Get the frame number for this Param. Initialized such that first
                                                   // update will set to 0, and increment at every update unless
                                                   // specified
        virtual Context* getContext() const = 0;   // Get the compute context of this Param
        virtual const std::shared_ptr<ICoordinateSystem>&
        getCoordinateSystem() const = 0;                 // Get the coordinate system of this Param
        virtual const TypeInfo& getTypeInfo() const = 0; // Implemented in concrete type
    };

    class MO_EXPORTS IParam : public ParamBase
    {
      public:
        typedef std::shared_ptr<IParam> Ptr;
        typedef std::shared_ptr<const IParam> ConstPtr;

        template <class... Args>
        IParam(const Args&... args)
        {
            _name = GetKeywordInputDefault<tag::param_name>("unnamed", args...);
            if (const mo::Time_t* ts = GetKeywordInputOptional<tag::timestamp>(args...))
                _ts = *ts;
            _mtx = nullptr;
            _fn = GetKeywordInputDefault<tag::frame_number>(-1, args...);
            _ctx = GetKeywordInputDefault<tag::context>(nullptr, args...);
            //_cs        = GetKeywordInputDefault<tag::coordinate_system>(nullptr, args...); // TODO keyword
            // specialization for std::shared_ptrs
            _flags = GetKeywordInputDefault<tag::param_flags>(EnumClassBitset<ParamFlags>(mo::ParamFlags::Control_e),
                                                              args...);
            _tree_root = GetKeywordInputDefault<tag::tree_root>("", args...);
        }

        IParam(const std::string& name_ = "",
               ParamFlags flags_ = ParamFlags::Control_e,
               OptionalTime_t ts_ = OptionalTime_t(),
               Context* ctx_ = nullptr,
               size_t fn_ = std::numeric_limits<size_t>::max());

        virtual ~IParam();

        IParam* setName(const std::string& name_); // Get the name of this Param
        IParam* setTreeRoot(
            const std::string& tree_root_); // Set the root to the name of this Param. IE objname:paramanme, set objname
        IParam* setContext(Context* ctx);   // Set the compute context of this Param
        IParam* setFrameNumber(size_t fn);  // Set the frame number for this Param
        IParam* setTimestamp(const mo::Time_t& ts); // Set the timestamp for this Param
        // Set the coordinate system for this Param
        IParam* setCoordinateSystem(const std::shared_ptr<ICoordinateSystem>& cs_);

        const std::string& getName() const; // Get the name of this Param
        const std::string
        getTreeName() const; // Get the name of this parmaeter appended with the tree root. IE root_name:param_name
        const std::string&
        getTreeRoot() const;                 // Get the tree root of this Param, ie the name of the owning parent object
        OptionalTime_t getTimestamp() const; // Get the timestamp of this Param, may not exist for all Params
        size_t getFrameNumber() const; // Get the frame number for this Param. Initialized such that first update will
                                       // set to 0, and increment at every update unless specified
        Context* getContext() const;   // Get the compute context of this Param
        const std::shared_ptr<ICoordinateSystem>&
        getCoordinateSystem() const; // Get the coordinate system of this Param

        void subscribe();              // Subscribe to this Param as an output
        void unsubscribe();            // unsubscribe to this Param as an output
        bool hasSubscriptions() const; // Determine if there are any input Params using this Param as an output

        // Register slots to be called on update of this Param
        std::shared_ptr<Connection> registerUpdateNotifier(UpdateSlot_t* f);
        std::shared_ptr<Connection> registerUpdateNotifier(std::shared_ptr<TSignalRelay<UpdateSig_t>>& relay);
        // Virtual to allow typed overload for interface slot input
        virtual std::shared_ptr<Connection> registerUpdateNotifier(ISlot* f);
        virtual std::shared_ptr<Connection> registerUpdateNotifier(std::shared_ptr<ISignalRelay> relay);

        // Register slots to be called on delete of this Param
        std::shared_ptr<Connection> registerDeleteNotifier(DeleteSlot_t* f);
        std::shared_ptr<Connection> registerDeleteNotifier(std::shared_ptr<TSignalRelay<void(IParam const*)>>& relay);
        std::shared_ptr<Connection> registerDeleteNotifier(ISlot* f);
        std::shared_ptr<Connection> registerDeleteNotifier(std::shared_ptr<ISignalRelay> relay);

        // commit changes to a Param, updates underlying meta info and emits signals accordingly
        virtual IParam* emitUpdate(
            const OptionalTime_t& ts_ = OptionalTime_t(),             // The timestamp of the new data
            Context* ctx_ = mo::Context::getCurrent(), // The context from which the data was updated
            const boost::optional<size_t>& fn_ = boost::optional<size_t>(), // The frame number of the update
            const std::shared_ptr<ICoordinateSystem>& cs_ =
                std::shared_ptr<ICoordinateSystem>(), // The coordinate system of the data
            UpdateFlags flags_ = ValueUpdated_e);

        virtual IParam*
        emitUpdate(const IParam& other); // commit a Param's value copying metadata info from another parmaeter
        template <class Archive>
        void serialize(Archive& ar); // Used for cereal serialization
        Mutex_t& mtx() const; // Get reference to Param mutex.  If setMtx was called, this will reference the mutex that
                              // was set, otherwise one will be created
        void setMtx(Mutex_t* mtx); // Use this to share a mutex with an owning object, ie a parent.

        EnumClassBitset<ParamFlags>
        appendFlags(ParamFlags flags_); // Append a flag to the Param, return previous values
        EnumClassBitset<ParamFlags>
        toggleFlags(ParamFlags flags_);                          // Toggle the value of a flag, returns previous flags
        bool checkFlags(ParamFlags flag) const;                  // Check if a single flag is set
        EnumClassBitset<ParamFlags> setFlags(ParamFlags flags_); // Set flags of the Param, return previous values
        EnumClassBitset<ParamFlags>
        setFlags(EnumClassBitset<ParamFlags> flags_); // Set flags of the Param, return previous values
        EnumClassBitset<ParamFlags> getFlags() const;
        bool modified() const;     // Check if has been modified
        void modified(bool value); // Set if it has been modified
      protected:
        template <class T>
        friend class UI::qt::ParamProxy;

        OptionalTime_t _ts;
        size_t _fn;
        std::shared_ptr<ICoordinateSystem> _cs;
        Context* _ctx;
        std::string _name;
        std::string _tree_root;
        mutable EnumClassBitset<ParamFlags> _flags;
        UpdateSignal_t _update_signal;
        DeleteSignal_t _delete_signal;
        mutable mo::Mutex_t* _mtx = nullptr;
        int _subscribers = 0;
        bool _modified = false; // Set to true if modified by the user interface etc, set to false by the owning object.
    };

    template <typename Archive>
    void IParam::serialize(Archive& ar)
    {
        ar(_name);
        ar(_tree_root);
        ar(_ts);
        ar(_fn);
        ar(_flags);
    }
}
