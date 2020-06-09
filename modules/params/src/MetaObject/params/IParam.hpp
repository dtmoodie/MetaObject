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
#include "ParamTags.hpp"
#include "TDataContainer.hpp"

#include <MetaObject/core/detail/forward.hpp>
#include <MetaObject/detail/Export.hpp>

#include <MetaObject/core.hpp>

#include <MetaObject/signals/TSignal.hpp>
#include <MetaObject/signals/TSlot.hpp>

#include <memory>
#include <string>

namespace mo
{
    struct ILoadVisitor;
    struct ISaveVisitor;
    struct IDataContainer;

    using UpdateSignal_t = TSignal<Update_s>;
    using DataUpdateSignal_t = TSignal<DataUpdate_s>;
    using UpdateSlotPtr_t = std::shared_ptr<UpdateSlot_t>;

    using DeleteSlot_t = TSlot<Delete_s>;
    using DeleteSignal_t = TSignal<Delete_s>;
    using DeleteSlotPtr_t = std::shared_ptr<DeleteSlot_t>;

    // need a pure virtual base class for python bindings

    struct MO_EXPORTS IParam
    {
        using Ptr_t = std::shared_ptr<IParam>;
        using ConstPtr_t = std::shared_ptr<const IParam>;

        IParam();
        IParam(const IParam&) = delete;
        IParam(IParam&&) = delete;
        IParam& operator=(const IParam&) = delete;
        IParam& operator=(IParam&&) = delete;

        virtual ~IParam();

        // Get the name of this param
        virtual void setName(std::string name_) = 0;

        // Set the root to the name of this param. IE objname:paramanme, set objname
        virtual void setTreeRoot(std::string tree_root_) = 0;

        // Set the compute stream of this param
        virtual void setStream(IAsyncStream& stream) = 0;

        // Get the name of this param
        virtual std::string getName() const = 0;

        // Get the name of this parmaeter appended with the tree root. IE root_name:param_name
        virtual std::string getTreeName() const = 0;

        // Get the tree root of this param, ie the name of the owning parent object
        virtual std::string getTreeRoot() const = 0;

        // Get the compute stream of this param
        virtual IAsyncStream* getStream() const = 0;

        // Virtual to allow typed overload for interface slot input
        virtual ConnectionPtr_t registerUpdateNotifier(ISlot& f) = 0;
        virtual ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr_t& relay) = 0;

        // Register slots to be called on delete of this param
        virtual ConnectionPtr_t registerDeleteNotifier(TSlot<Delete_s>& f) = 0;
        virtual ConnectionPtr_t registerDeleteNotifier(const TSignalRelay<Delete_s>::Ptr_t& relay) = 0;

        // Get reference to param mutex.  If setMtx was called, this will reference the mutex that
        // was set, otherwise one will be created
        virtual Mutex_t& mtx() const = 0;

        // Use this to share a mutex with an owning object, ie a parent.
        virtual void setMtx(Mutex_t& mtx) = 0;

        // Append a flag to the Param, return previous values
        virtual ParamFlags appendFlags(ParamFlags flags_) = 0;

        // Check if a single flag is set
        virtual bool checkFlags(ParamFlags flag) const = 0;

        // Set flags of the param, return previous values
        virtual ParamFlags setFlags(ParamFlags flags_) = 0;

        virtual ParamFlags getFlags() const = 0;

        virtual std::ostream& print(std::ostream& os) const = 0;

        virtual void load(ILoadVisitor&) = 0;
        virtual void save(ISaveVisitor&) const = 0;
        virtual void visit(StaticVisitor&) const = 0;

        virtual void setLogger(spdlog::logger& logger) = 0;
    };
} // namespace mo
