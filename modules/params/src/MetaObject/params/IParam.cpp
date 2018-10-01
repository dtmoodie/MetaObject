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
#include "MetaObject/params/IParam.hpp"
#include "ICoordinateSystem.hpp"
#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"
#include "MetaObject/signals/TSlot.hpp"
#include <algorithm>
#include <boost/thread/recursive_mutex.hpp>
namespace mo
{

    ParamBase::~ParamBase()
    {
    }

    namespace kwargs
    {
        TKeyword<tag::param> TKeyword<tag::param>::instance;

        TaggedArgument<tag::param> TKeyword<tag::param>::operator=(const IParam& data)
        {
            return TaggedArgument<tag::param>(data);
        }

        template MO_EXPORTS struct TKeyword<tag::param>;
    }

    IParam::IParam(const std::string& name_, ParamFlags flags_, Context* ctx)
        : _name(name_)
        , _flags(flags_)
        , _mtx(nullptr)
        , _subscribers(0)
        , _modified(false)
    {
        _header.frame_number = std::numeric_limits<uint64_t>::max();
        _header.ctx = ctx;
    }

    IParam::~IParam()
    {
        _delete_signal(this);
        if (checkFlags(ParamFlags::OwnsMutex_e))
            delete _mtx;
    }

    IParam* IParam::setName(const std::string& name_)
    {
        _name = name_;
        return this;
    }

    IParam* IParam::setTreeRoot(const std::string& treeRoot_)
    {
        _tree_root = treeRoot_;
        return this;
    }

    IParam* IParam::setFrameNumber(const uint64_t fn)
    {
        _header.frame_number = fn;
        return this;
    }

    IParam* IParam::setTimestamp(const mo::Time_t& ts)
    {
        _header.timestamp = ts;
        return this;
    }

    IParam* IParam::setCoordinateSystem(const std::shared_ptr<ICoordinateSystem>& system)
    {
        _header.coordinate_system = system;
        return this;
    }

    IParam* IParam::setContext(Context* ctx)
    {
        _header.ctx = ctx;
        return this;
    }

    const std::string& IParam::getName() const
    {
        return _name;
    }

    const std::string& IParam::getTreeRoot() const
    {
        return _tree_root;
    }

    const std::string IParam::getTreeName() const
    {
        if (_tree_root.size())
        {
            return _tree_root + ":" + _name;
        }
        else
        {
            return _name;
        }
    }

    OptionalTime_t IParam::getTimestamp() const
    {
        return _header.timestamp;
    }

    uint64_t IParam::getFrameNumber() const
    {
        return _header.frame_number;
    }

    Context* IParam::getContext() const
    {
        return _header.ctx;
    }

    const std::shared_ptr<ICoordinateSystem>& IParam::getCoordinateSystem() const
    {
        return _header.coordinate_system;
    }

    Header IParam::getHeader() const
    {
        return _header;
    }

    std::shared_ptr<Connection> IParam::registerUpdateNotifier(UpdateSlot_t* f)
    {
        Lock lock(mtx());
        return f->connect(&_update_signal);
    }

    std::shared_ptr<Connection> IParam::registerUpdateNotifier(ISlot* f)
    {
        Lock lock(mtx());
        auto typed = dynamic_cast<UpdateSlot_t*>(f);
        if (typed)
        {
            return registerUpdateNotifier(typed);
        }
        return std::shared_ptr<Connection>();
    }

    std::shared_ptr<Connection> IParam::registerUpdateNotifier(std::shared_ptr<ISignalRelay> relay)
    {
        Lock lock(mtx());
        auto typed = std::dynamic_pointer_cast<TSignalRelay<Update_s>>(relay);
        if (typed)
        {
            return registerUpdateNotifier(typed);
        }
        return std::shared_ptr<Connection>();
    }

    std::shared_ptr<Connection> IParam::registerUpdateNotifier(TSignalRelay<Update_s>::Ptr& relay)
    {
        Lock lock(mtx());
        return _update_signal.connect(relay);
    }

    std::shared_ptr<Connection> IParam::registerDeleteNotifier(DeleteSlot_t* f)
    {
        Lock lock(mtx());
        return f->connect(&_delete_signal);
    }

    std::shared_ptr<Connection> IParam::registerDeleteNotifier(ISlot* f)
    {
        Lock lock(mtx());
        auto typed = dynamic_cast<DeleteSlot_t*>(f);
        if (typed)
        {
            return registerDeleteNotifier(typed);
        }
        return std::shared_ptr<Connection>();
    }

    std::shared_ptr<Connection> IParam::registerDeleteNotifier(ISignalRelay::Ptr relay)
    {
        Lock lock(mtx());
        auto typed = std::dynamic_pointer_cast<TSignalRelay<void(IParam*)>>(relay);
        if (typed)
        {
            return registerDeleteNotifier(typed);
        }
        return std::shared_ptr<Connection>();
    }

    std::shared_ptr<Connection> IParam::registerDeleteNotifier(TSignalRelay<void(IParam const*)>::Ptr& relay)
    {
        Lock lock(mtx());
        return _delete_signal.connect(relay);
    }

    IParam* IParam::emitUpdate(const Header& header, UpdateFlags flags_)
    {
        Lock lock(mtx());
        uint64_t fn = header.frame_number;
        if (fn == std::numeric_limits<uint64_t>::max())
        {
            fn = _header.frame_number + 1;
        }
        _header = header;
        _header.frame_number = fn;
        this->modified(true);
        lock.unlock();
        _update_signal(this, _header, flags_);
        return this;
    }

    IParam* IParam::emitUpdate(const IParam& other, UpdateFlags flags_)
    {
        return emitUpdate(other._header, flags_);
        return this;
    }

    Mutex_t& IParam::mtx() const
    {
        if (_mtx == nullptr)
        {
            _mtx = new boost::recursive_timed_mutex();
            _flags.set(ParamFlags::OwnsMutex_e);
        }
        return *_mtx;
    }

    void IParam::setMtx(boost::recursive_timed_mutex* mtx_)
    {
        if (_mtx && checkFlags(ParamFlags::OwnsMutex_e))
        {
            delete _mtx;
            _flags.reset(ParamFlags::OwnsMutex_e);
        }
        _mtx = mtx_;
    }

    void IParam::subscribe()
    {
        Lock lock(mtx());
        ++_subscribers;
    }

    void IParam::unsubscribe()
    {
        Lock lock(mtx());
        --_subscribers;
        _subscribers = std::max(0, _subscribers);
    }

    bool IParam::hasSubscriptions() const
    {
        return _subscribers != 0;
    }

    EnumClassBitset<ParamFlags> IParam::setFlags(ParamFlags flags_)
    {
        auto prev = _flags;
        _flags.set(flags_);
        return prev;
    }

    EnumClassBitset<ParamFlags> IParam::setFlags(EnumClassBitset<ParamFlags> flags_)
    {
        auto prev = _flags;
        _flags = flags_;
        return prev;
    }

    EnumClassBitset<ParamFlags> IParam::appendFlags(ParamFlags flags_)
    {
        auto prev = _flags;
        _flags.set(flags_);
        return prev;
    }

    bool IParam::checkFlags(ParamFlags flag) const
    {
        return _flags.test(flag);
    }

    EnumClassBitset<ParamFlags> IParam::getFlags() const
    {
        return _flags;
    }

    bool IParam::modified() const
    {
        return _modified;
    }

    void IParam::modified(bool value)
    {
        _modified = value;
    }

    std::ostream& IParam::print(std::ostream& os) const
    {
        os << getTreeName();
        os << " [" << mo::Demangle::typeToName(getTypeInfo()) << "]";
        auto ts = getTimestamp();
        if (ts)
        {
            os << " " << *ts;
        }
        auto fn = getFrameNumber();
        if (fn != std::numeric_limits<uint64_t>::max())
        {
            os << " " << fn;
        }
        auto cs = getCoordinateSystem();
        if (cs)
        {
            os << " " << cs->getName();
        }
        auto ctx = getContext();
        if (ctx)
        {
            os << " <" << ctx->getName() << ">";
        }
        return os;
    }

} // namespace mo
