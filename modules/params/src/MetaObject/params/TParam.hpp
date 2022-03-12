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
#ifndef MO_PARAMS_TPARAM_HPP
#define MO_PARAMS_TPARAM_HPP
#include <MetaObject/detail/Export.hpp>

#include <MetaObject/params/IControlParam.hpp>
#include <MetaObject/params/IPublisher.hpp>
#include <MetaObject/params/ISubscriber.hpp>

#include <string>

namespace mo
{

    template <class BASE>
    struct MO_EXPORTS TParam : BASE
    {
        ~TParam() override;
        void setName(std::string name) override;
        void setTreeRoot(std::string tree_root) override;
        void setStream(IAsyncStream& stream) override;
        std::string getName() const override;
        std::string getTreeName() const override;
        std::string getTreeRoot() const override;
        IAsyncStream* getStream() const override;

        ConnectionPtr_t registerUpdateNotifier(ISlot& f) override;
        ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr_t& relay) override;
        ConnectionPtr_t registerDeleteNotifier(TSlot<Delete_s>& f) override;
        ConnectionPtr_t registerDeleteNotifier(const TSignalRelay<Delete_s>::Ptr_t& relay) override;
        Mutex_t& mtx() const override;
        void setMtx(Mutex_t& mtx) override;
        ParamFlags appendFlags(ParamFlags flags) override;
        bool checkFlags(ParamFlags flag) const override;
        ParamFlags setFlags(ParamFlags flags) override;
        ParamFlags getFlags() const override;
        void setLogger(spdlog::logger& logger) override;

        void load(ILoadVisitor&) override;
        void save(ISaveVisitor&) const override;
        void visit(StaticVisitor&) const override;

      protected:
        spdlog::logger& getLogger() const;

        void emitUpdate(Header, UpdateFlags, IAsyncStream*) const;

      private:
        mutable mo::Mutex_t* m_mtx = nullptr;
        IAsyncStream* m_stream = nullptr;
        spdlog::logger* m_logger = nullptr;

        std::string m_name;
        std::string m_root_name;

        ParamFlags m_flags;
        bool m_modified = false;

        TSignal<Update_s> m_update_signal;
        TSignal<Delete_s> m_delete_signal;

        mutable std::unique_ptr<Mutex_t> m_mtx_ptr;
    };
#ifdef MO_TEMPLATE_EXTERN
    extern template struct TParam<ISubscriber>;
    extern template struct TParam<IPublisher>;
#endif // MO_TEMPLATE_EXTERN
    // extern template struct TParam<IControlParam>;

    /////////////////////////////////////////////////////////////////////////

    template <class BASE>
    TParam<BASE>::~TParam()
    {
        m_delete_signal(*this);
    }

    template <class BASE>
    void TParam<BASE>::setName(std::string name)
    {
        Mutex_t::Lock_t lock(mtx());
        m_name = std::move(name);
    }

    template <class BASE>
    void TParam<BASE>::setTreeRoot(std::string tree_root)
    {
        Mutex_t::Lock_t lock(mtx());
        m_root_name = std::move(tree_root);
    }

    template <class BASE>
    void TParam<BASE>::setStream(IAsyncStream& stream)
    {
        Mutex_t::Lock_t lock(mtx());
        m_stream = &stream;
    }

    template <class BASE>
    std::string TParam<BASE>::getName() const
    {
        Mutex_t::Lock_t lock(mtx());
        return m_name;
    }

    template <class BASE>
    std::string TParam<BASE>::getTreeName() const
    {
        Mutex_t::Lock_t lock(mtx());
        if (m_root_name.empty())
        {
            return m_name;
        }
        return m_root_name + ':' + m_name;
    }

    template <class BASE>
    std::string TParam<BASE>::getTreeRoot() const
    {
        Mutex_t::Lock_t lock(mtx());
        return m_root_name;
    }

    template <class BASE>
    IAsyncStream* TParam<BASE>::getStream() const
    {
        return m_stream;
    }

    template <class BASE>
    ConnectionPtr_t TParam<BASE>::registerUpdateNotifier(ISlot& f)
    {
        return m_update_signal.connect(f);
    }

    template <class BASE>
    ConnectionPtr_t TParam<BASE>::registerUpdateNotifier(const ISignalRelay::Ptr_t& relay_)
    {
        if (relay_ == nullptr)
        {
            return {};
        }
        auto relay = relay_;
        return m_update_signal.connect(relay);
    }

    template <class BASE>
    ConnectionPtr_t TParam<BASE>::registerDeleteNotifier(TSlot<Delete_s>& f)
    {
        return m_delete_signal.connect(f);
    }

    template <class BASE>
    ConnectionPtr_t TParam<BASE>::registerDeleteNotifier(const TSignalRelay<Delete_s>::Ptr_t& relay)
    {
        if (relay == nullptr)
        {
            return {};
        }
        auto tmp = relay;
        return m_delete_signal.connect(tmp);
    }
    template <class BASE>
    Mutex_t& TParam<BASE>::mtx() const
    {
        if (!m_mtx)
        {
            m_mtx_ptr.reset(new Mutex_t());
            m_mtx = m_mtx_ptr.get();
        }
        return *m_mtx;
    }

    template <class BASE>
    void TParam<BASE>::setMtx(Mutex_t& mtx)
    {
        if (m_mtx)
        {
            m_mtx_ptr.reset();
            m_mtx = nullptr;
        }
        m_mtx = &mtx;
    }

    template <class BASE>
    ParamFlags TParam<BASE>::appendFlags(ParamFlags flags_)
    {
        Mutex_t::Lock_t lock(mtx());
        auto prev = m_flags;
        m_flags.set(flags_);
        return prev;
    }

    template <class BASE>
    bool TParam<BASE>::checkFlags(ParamFlags flag) const
    {
        Mutex_t::Lock_t lock(mtx());
        return m_flags.test(flag);
    }

    template <class BASE>
    ParamFlags TParam<BASE>::setFlags(ParamFlags flags_)
    {
        Mutex_t::Lock_t lock(mtx());
        auto prev = m_flags;
        m_flags = flags_;
        return prev;
    }

    template <class BASE>
    ParamFlags TParam<BASE>::getFlags() const
    {
        Mutex_t::Lock_t lock(mtx());
        return m_flags;
    }

    template <class BASE>
    void TParam<BASE>::setLogger(spdlog::logger& logger)
    {
        Mutex_t::Lock_t lock(mtx());
        m_logger = &logger;
    }

    template <class BASE>
    void TParam<BASE>::load(ILoadVisitor& visitor)
    {
        visitor(&m_name, "name");
        visitor(&m_root_name, "root_name");
        visitor(&m_flags, "flags");
    }

    template <class BASE>
    void TParam<BASE>::save(ISaveVisitor& visitor) const
    {
        visitor(&m_name, "name");
        visitor(&m_root_name, "root_name");
        visitor(&m_flags, "flags");
    }

    template <class BASE>
    void TParam<BASE>::visit(StaticVisitor& visitor) const
    {
        visitor.template visit<std::string>("name");
        visitor.template visit<std::string>("root_name");
        visitor.template visit<ParamFlags>("flags");
    }

    template <class BASE>
    spdlog::logger& TParam<BASE>::getLogger() const
    {
        Mutex_t::Lock_t lock(mtx());
        if (!m_logger)
        {
            return getDefaultLogger();
        }
        MO_ASSERT(m_logger != nullptr);
        return *m_logger;
    }

    template <class BASE>
    void TParam<BASE>::emitUpdate(Header hdr, UpdateFlags fgs, IAsyncStream* stream) const
    {
        m_update_signal(*this, std::move(hdr), std::move(fgs), stream);
    }

} // namespace mo
#endif // MO_PARAMS_TPARAM_HPP
