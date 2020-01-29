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
#include "Header.hpp"
#include "MetaObject/core.hpp"
#include "MetaObject/core/detail/forward.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "ParamTags.hpp"
#include "TDataContainer.hpp"
#include <MetaObject/detail/Export.hpp>
#include <string>

namespace mo
{
    struct ILoadVisitor;
    struct ISaveVisitor;
    struct IDataContainer;

    using UpdateSignal_t = TSignal<Update_s>;
    using DataUpdateSignal_t = TSignal<DataUpdate_s>;
    using UpdateSlotPtr_t = std::shared_ptr<UpdateSlot_t>;

    using DeleteSlot_t = TSlot<void(const IParam*)>;
    using DeleteSignal_t = TSignal<void(const IParam*)>;
    using DeleteSlotPtr_t = std::shared_ptr<DeleteSlot_t>;

    // need a pure virtual base class for python bindings
    class MO_EXPORTS ParamBase
    {
      public:
        using Ptr_t = std::shared_ptr<ParamBase>;
        using ConstPtr_t = std::shared_ptr<const ParamBase>;
        using IContainerPtr_t = std::shared_ptr<IDataContainer>;
        using IContainerConstPtr_t = std::shared_ptr<const IDataContainer>;

        ParamBase() = default;
        ParamBase(const ParamBase&) = delete;
        ParamBase& operator=(const ParamBase&) = delete;
        ParamBase(ParamBase&&) = delete;
        ParamBase& operator=(ParamBase&&) = delete;

        virtual ~ParamBase();

        // Get the name of this param
        virtual const std::string& getName() const = 0;

        // Get the name of this parmaeter appended with the tree root. IE root_name:param_name
        virtual const std::string getTreeName() const = 0;

        // Get the tree root of this param, ie the name of the owning parent object
        virtual const std::string& getTreeRoot() const = 0;

        // Get the timestamp of this param, may not exist for all params
        virtual OptionalTime getTimestamp() const = 0;

        // Get the frame number for this param. Initialized such that first
        // update will set to 0, and increment at every update unless
        // specified
        virtual FrameNumber getFrameNumber() const = 0;

        // Get the compute stream of this param
        virtual IAsyncStream* getStream() const = 0;

        // Get the coordinate system of this param
        virtual const ICoordinateSystemPtr_t& getCoordinateSystem() const = 0;

        // Implemented in concrete type
        virtual TypeInfo getTypeInfo() const = 0;
        virtual std::ostream& print(std::ostream& os) const = 0;

        // Append a flag to the Param, return previous values
        virtual ParamFlags appendFlags(ParamFlags flags_) = 0;

        // Check if a single flag is set
        virtual bool checkFlags(ParamFlags flag) const = 0;

        // Set flags of the param, return previous values
        virtual ParamFlags setFlags(ParamFlags flags_) = 0;

        virtual ParamFlags getFlags() const = 0;

        virtual void load(ILoadVisitor&) = 0;
        virtual void save(ISaveVisitor&) const = 0;
        virtual void load(BinaryInputVisitor& ar) = 0;
        virtual void save(BinaryOutputVisitor& ar) const = 0;
        virtual void visit(StaticVisitor&) const = 0;

        virtual IContainerPtr_t getData(const Header& desired = Header()) = 0;
        virtual IContainerConstPtr_t getData(const Header& desired = Header()) const = 0;

        template <class T>
        std::shared_ptr<TDataContainer<T>> getTypedData(const Header& desired = Header());
        template <class T>
        std::shared_ptr<const TDataContainer<T>> getTypedData(const Header& desired = Header()) const;

        template <class T>
        bool getTypedData(T*, const Header& desired = Header(), Header* retrieved = nullptr) const;
        template <class T>
        bool getTypedData(T*, const Header& desired = Header(), Header* retrieved = nullptr);
    };

    class MO_EXPORTS IParam : public ParamBase
    {
      public:
        using Ptr_t = std::shared_ptr<IParam>;
        using ConstPtr_t = std::shared_ptr<const IParam>;

        IParam(const IParam&) = delete;
        IParam(IParam&&) = delete;
        IParam& operator=(const IParam&) = delete;
        IParam& operator=(IParam&&) = delete;

        IParam(const std::string& name_ = "", ParamFlags flags_ = ParamFlags::kCONTROL, IAsyncStream* ctx_ = nullptr);

        ~IParam() override;

        // Get the name of this param
        IParam* setName(const std::string& name_);

        // Set the root to the name of this param. IE objname:paramanme, set objname
        IParam* setTreeRoot(const std::string& tree_root_);

        // Set the compute stream of this param
        virtual IParam* setStream(IAsyncStream* ctx);

        // Set the coordinate system for this param
        IParam* setCoordinateSystem(const std::shared_ptr<ICoordinateSystem>& cs_);

        // Get the name of this param
        const std::string& getName() const override;

        // Get the name of this parmaeter appended with the tree root. IE root_name:param_name
        const std::string getTreeName() const override;

        // Get the tree root of this param, ie the name of the owning parent object
        const std::string& getTreeRoot() const override;

        // Get the compute stream of this param
        IAsyncStream* getStream() const override;

        // Get the coordinate system of this Param
        const ICoordinateSystemPtr_t& getCoordinateSystem() const override;

        // Subscribe to this param as an output
        void subscribe();
        // unsubscribe to this param as an output
        void unsubscribe();
        // Determine if there are any input Params using this Param as an output
        bool hasSubscriptions() const;

        // Virtual to allow typed overload for interface slot input
        virtual ConnectionPtr_t registerUpdateNotifier(ISlot* f);
        virtual ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr_t& relay);

        // Register slots to be called on delete of this param
        virtual ConnectionPtr_t registerDeleteNotifier(ISlot* f);
        virtual ConnectionPtr_t registerDeleteNotifier(const ISignalRelay::Ptr_t& relay);

        // commit changes to a param, updates underlying meta info and emits signals accordingly
        virtual IParam* emitUpdate(const Header& header = Header(), UpdateFlags flags_ = UpdateFlags::kVALUE_UPDATED);
        virtual IParam* emitUpdate(const IDataContainerPtr_t& data, UpdateFlags flags = UpdateFlags::kVALUE_UPDATED);

        // commit a param's value copying metadata info from another parmaeter
        virtual IParam* emitUpdate(const IParam& other, UpdateFlags flags_ = UpdateFlags::kVALUE_UPDATED);

        // Get reference to param mutex.  If setMtx was called, this will reference the mutex that
        // was set, otherwise one will be created
        Mutex_t& mtx() const;

        // Use this to share a mutex with an owning object, ie a parent.
        virtual void setMtx(Mutex_t* mtx);

        // Append a flag to the Param, return previous values
        ParamFlags appendFlags(ParamFlags flags_) override;

        // Check if a single flag is set
        bool checkFlags(ParamFlags flag) const override;

        // Set flags of the param, return previous values
        ParamFlags setFlags(ParamFlags flags_) override;

        ParamFlags getFlags() const override;

        // Check if has been modified
        virtual bool modified() const;

        // Set if it has been modified
        virtual void modified(bool value);

        std::ostream& print(std::ostream& os) const override;

      protected:
        std::string m_name;
        std::string m_tree_root;
        mutable ParamFlags m_flags;

      private:
        UpdateSignal_t m_update_signal;
        DataUpdateSignal_t m_data_update;
        DeleteSignal_t m_delete_signal;
        // Set to true if modified by the user interface etc, set to false by the owning object.
        bool m_modified = false;
        int m_subscribers = 0;
        mutable Mutex_t* m_mtx = nullptr;

        IAsyncStream* m_stream;
        std::shared_ptr<ICoordinateSystem> m_cs;
    };

    template <class T>
    std::shared_ptr<TDataContainer<T>> ParamBase::getTypedData(const Header& desired)
    {
        auto data = getData(desired);
        if (data)
        {
            return std::static_pointer_cast<TDataContainer<T>>(data);
        }
        return {};
    }

    template <class T>
    std::shared_ptr<const TDataContainer<T>> ParamBase::getTypedData(const Header& desired) const
    {
        auto data = getData(desired);
        if (data)
        {
            return std::static_pointer_cast<const TDataContainer<T>>(data);
        }
        return {};
    }
    template <class T>
    bool ParamBase::getTypedData(T* data, const Header& desired, Header* retrieved) const
    {
        auto container = getTypedData<T>(desired);
        if (container)
        {
            if (retrieved)
            {
                *retrieved = container->getHeader();
            }
            *data = container->data;
            return true;
        }
        return false;
    }

    template <class T>
    bool ParamBase::getTypedData(T* data, const Header& desired, Header* retrieved)
    {
        auto container = getTypedData<T>(desired);
        if (container)
        {
            if (retrieved)
            {
                *retrieved = container->getHeader();
            }
            *data = container->data;
            return true;
        }
        return false;
    }
}
