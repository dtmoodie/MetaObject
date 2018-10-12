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
#include "IDataContainer.hpp"
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
        : m_name(name_)
        , m_flags(flags_)
        , m_subscribers(0)
        , m_modified(false)
        , m_mtx(nullptr)
    {
        m_header.frame_number = std::numeric_limits<uint64_t>::max();
        m_header.ctx = ctx;
    }

    IParam::~IParam()
    {
        m_delete_signal(this);
        if (checkFlags(ParamFlags::OwnsMutex_e))
        {
            delete m_mtx;
        }
    }

    IParam* IParam::setName(const std::string& name_)
    {
        m_name = name_;
        return this;
    }

    IParam* IParam::setTreeRoot(const std::string& treeRoot_)
    {
        m_tree_root = treeRoot_;
        return this;
    }

    IParam* IParam::setFrameNumber(const uint64_t fn)
    {
        m_header.frame_number = fn;
        return this;
    }

    IParam* IParam::setTimestamp(const mo::Time& ts)
    {
        m_header.timestamp = ts;
        return this;
    }

    IParam* IParam::setCoordinateSystem(const std::shared_ptr<ICoordinateSystem>& system)
    {
        m_header.coordinate_system = system;
        return this;
    }

    IParam* IParam::setContext(Context* ctx)
    {
        m_ctx = ctx;
        return this;
    }

    const std::string& IParam::getName() const
    {
        return m_name;
    }

    const std::string& IParam::getTreeRoot() const
    {
        return m_tree_root;
    }

    const std::string IParam::getTreeName() const
    {
        if (m_tree_root.size())
        {
            return m_tree_root + ":" + m_name;
        }
        else
        {
            return m_name;
        }
    }

    OptionalTime IParam::getTimestamp() const
    {
        return m_header.timestamp;
    }

    uint64_t IParam::getFrameNumber() const
    {
        return m_header.frame_number;
    }

    Context* IParam::getContext() const
    {
        return m_ctx;
    }

    const std::shared_ptr<ICoordinateSystem>& IParam::getCoordinateSystem() const
    {
        return m_header.coordinate_system;
    }

    Header IParam::getHeader() const
    {
        return m_header;
    }

    ConnectionPtr_t IParam::registerUpdateNotifier(ISlot* f)
    {
        Lock lock(mtx());
        if (f->getSignature() == m_data_update.getSignature())
        {
            return m_data_update.connect(f);
        }
        if (f->getSignature() == m_update_signal.getSignature())
        {
            return m_update_signal.connect(f);
        }
        return {};
    }

    ConnectionPtr_t IParam::registerUpdateNotifier(const ISignalRelay::Ptr& relay)
    {
        Lock lock(mtx());
        auto tmp = relay;
        if (relay->getSignature() == m_data_update.getSignature())
        {
            return m_data_update.connect(tmp);
        }
        if (relay->getSignature() == m_update_signal.getSignature())
        {
            return m_update_signal.connect(tmp);
        }
        return {};
    }

    ConnectionPtr_t IParam::registerDeleteNotifier(ISlot* f)
    {
        Lock lock(mtx());
        if (m_delete_signal.getSignature() == f->getSignature())
        {
            return m_delete_signal.connect(f);
        }
        return {};
    }

    ConnectionPtr_t IParam::registerDeleteNotifier(const ISignalRelay::Ptr& relay)
    {
        Lock lock(mtx());
        if (relay->getSignature() == m_delete_signal.getSignature())
        {
            auto tmp = relay;
            m_delete_signal.connect(tmp);
        }
        return {};
    }

    IParam* IParam::emitUpdate(const Header& header, UpdateFlags flags_)
    {
        Lock lock(mtx());
        uint64_t fn = header.frame_number;
        if (fn == std::numeric_limits<uint64_t>::max())
        {
            fn = m_header.frame_number + 1;
        }
        m_header = header;
        m_header.frame_number = fn;
        modified(true);
        lock.unlock();
        m_update_signal(this, m_header, flags_);
        return this;
    }

    IParam* IParam::emitUpdate(const IDataContainerPtr_t& data, UpdateFlags flags)
    {
        Lock lock(mtx());
        uint64_t fn = data->getHeader().frame_number;
        if (fn == std::numeric_limits<uint64_t>::max())
        {
            fn = m_header.frame_number + 1;
        }
        m_header = data->getHeader();
        m_header.frame_number = fn;
        modified(true);
        lock.unlock();
        m_data_update(data, this, flags);
        m_update_signal(this, m_header, flags);
        return this;
    }

    IParam* IParam::emitUpdate(const IParam& other, UpdateFlags flags_)
    {
        return emitUpdate(other.m_header, flags_);
        return this;
    }

    Mutex_t& IParam::mtx() const
    {
        if (m_mtx == nullptr)
        {
            m_mtx = new boost::recursive_timed_mutex();
            m_flags.set(ParamFlags::OwnsMutex_e);
        }
        return *m_mtx;
    }

    void IParam::setMtx(boost::recursive_timed_mutex* mtx_)
    {
        if (m_mtx && checkFlags(ParamFlags::OwnsMutex_e))
        {
            delete m_mtx;
            m_flags.reset(ParamFlags::OwnsMutex_e);
        }
        m_mtx = mtx_;
    }

    void IParam::subscribe()
    {
        Lock lock(mtx());
        ++m_subscribers;
    }

    void IParam::unsubscribe()
    {
        Lock lock(mtx());
        --m_subscribers;
        m_subscribers = std::max(0, m_subscribers);
    }

    bool IParam::hasSubscriptions() const
    {
        return m_subscribers != 0;
    }

    EnumClassBitset<ParamFlags> IParam::setFlags(ParamFlags flags_)
    {
        auto prev = m_flags;
        m_flags.set(flags_);
        return prev;
    }

    EnumClassBitset<ParamFlags> IParam::setFlags(EnumClassBitset<ParamFlags> flags_)
    {
        auto prev = m_flags;
        m_flags = flags_;
        return prev;
    }

    EnumClassBitset<ParamFlags> IParam::appendFlags(ParamFlags flags_)
    {
        auto prev = m_flags;
        m_flags.set(flags_);
        return prev;
    }

    bool IParam::checkFlags(ParamFlags flag) const
    {
        return m_flags.test(flag);
    }

    EnumClassBitset<ParamFlags> IParam::getFlags() const
    {
        return m_flags;
    }

    bool IParam::modified() const
    {
        return m_modified;
    }

    void IParam::modified(bool value)
    {
        m_modified = value;
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
