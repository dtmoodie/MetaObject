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
#include "MetaObject/core/TypeTable.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"
#include "MetaObject/signals/TSlot.hpp"

#include <algorithm>

#include <MetaObject/thread/fiber_include.hpp>

#include <boost/thread/locks.hpp>

namespace mo
{

    ParamBase::~ParamBase()
    {
    }

    namespace kwargs
    {
        
    }

    IParam::IParam(const std::string& name_, ParamFlags flags_, IAsyncStream* stream)
        : m_name(name_)
        , m_flags(flags_)
        , m_stream(stream)
    {
    }

    IParam::~IParam()
    {
        m_delete_signal(this);
        if (checkFlags(ParamFlags::kOWNS_MUTEX))
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

    IParam* IParam::setCoordinateSystem(const std::shared_ptr<ICoordinateSystem>& system)
    {
        m_cs = system;
        return this;
    }

    IParam* IParam::setStream(IAsyncStream* stream)
    {
        m_stream = stream;
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
        if (!m_tree_root.empty())
        {
            return m_tree_root + ":" + m_name;
        }
        return m_name;
    }

    IAsyncStream* IParam::getStream() const
    {
        return m_stream;
    }

    const std::shared_ptr<ICoordinateSystem>& IParam::getCoordinateSystem() const
    {
        return m_cs;
    }

    ConnectionPtr_t IParam::registerUpdateNotifier(ISlot* f)
    {
        Lock_t lock(mtx());
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

    ConnectionPtr_t IParam::registerUpdateNotifier(const ISignalRelay::Ptr_t& relay)
    {
        Lock_t lock(mtx());
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
        Lock_t lock(mtx());
        if (m_delete_signal.getSignature() == f->getSignature())
        {
            return m_delete_signal.connect(f);
        }
        return {};
    }

    ConnectionPtr_t IParam::registerDeleteNotifier(const ISignalRelay::Ptr_t& relay)
    {
        Lock_t lock(mtx());
        if (relay->getSignature() == m_delete_signal.getSignature())
        {
            auto tmp = relay;
            m_delete_signal.connect(tmp);
        }
        return {};
    }

    IParam* IParam::emitUpdate(const Header& header, UpdateFlags flags_)
    {
        modified(true);
        m_update_signal(this, header, flags_);
        return this;
    }

    IParam* IParam::emitUpdate(const IDataContainerPtr_t& data, UpdateFlags flags)
    {
        modified(true);
        m_data_update(data, this, flags);
        m_update_signal(this, data->getHeader(), flags);
        return this;
    }

    IParam* IParam::emitUpdate(const IParam& other, UpdateFlags flags_)
    {
        auto ts = other.getTimestamp();
        auto fn = other.getFrameNumber();

        return emitUpdate(mo::Header(ts, fn), flags_);
    }

    Mutex_t& IParam::mtx() const
    {
        if (m_mtx == nullptr)
        {
            m_mtx = new Mutex_t();
            m_flags.set(ParamFlags::kOWNS_MUTEX);
        }
        return *m_mtx;
    }

    void IParam::setMtx(Mutex_t* mtx_)
    {
        if (m_mtx && checkFlags(ParamFlags::kOWNS_MUTEX))
        {
            delete m_mtx;
            m_flags.reset(ParamFlags::kOWNS_MUTEX);
        }
        m_mtx = mtx_;
    }

    void IParam::subscribe()
    {
        Lock_t lock(mtx());
        ++m_subscribers;
    }

    void IParam::unsubscribe()
    {
        Lock_t lock(mtx());
        --m_subscribers;
        m_subscribers = std::max(0, m_subscribers);
    }

    bool IParam::hasSubscriptions() const
    {
        return m_subscribers != 0;
    }

    ParamFlags IParam::setFlags(ParamFlags flags_)
    {
        auto prev = m_flags;
        m_flags = flags_;
        return prev;
    }

    ParamFlags IParam::appendFlags(ParamFlags flags_)
    {
        auto prev = m_flags;
        m_flags.set(flags_);
        return prev;
    }

    bool IParam::checkFlags(ParamFlags flag) const
    {
        return m_flags.test(flag);
    }

    ParamFlags IParam::getFlags() const
    {
        return m_flags;
    }

    bool IParam::modified() const
    {
        return m_modified;
    }

    void IParam::modified(bool value)
    {
        Lock_t lock(mtx());
        m_modified = value;
    }

    std::ostream& IParam::print(std::ostream& os) const
    {
        os << getTreeName();
        os << " [" << mo::TypeTable::instance()->typeToName(getTypeInfo()) << "]";
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
        auto stream = getStream();
        if (stream)
        {
            os << " <" << stream->name() << ">";
        }
        return os;
    }

} // namespace mo
