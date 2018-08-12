#ifndef TMULTIINPUTINL_HPP
#define TMULTIINPUTINL_HPP
#include "TMultiInput.hpp"

namespace mo
{

    template <class... Types>
    TMultiInput<Types...>::TMultiInput() : InputParam(), IParam("", mo::ParamFlags::Input_e)
    {
        this->setFlags(mo::ParamFlags::Input_e);
    }

    template <class... Types>
    typename TMultiInput<Types...>::InputTypeTuple TMultiInput<Types...>::initNullptr()
    {
        InputTypeTuple out;
        Initializer init;
        typeLoop<Types...>(init, out);
        return out;
    }

    template <class... Types>
    void TMultiInput<Types...>::setUserDataPtr(std::tuple<const Types*...>* user_var_)
    {
        typeLoop<Types...>(*this, user_var_);
    }

    template <class... Types>
    bool TMultiInput<Types...>::setInput(std::shared_ptr<IParam> input)
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        bool success = false;
        if (m_current_input)
        {
            selectType<Types...>(*this, m_current_input->getTypeInfo(), static_cast<IParam*>(nullptr), &success);
            success = false;
        }
        selectType<Types...>(*this, input->getTypeInfo(), input, &success);
        if (success)
        {
            m_current_input = input.get();
        }
        return success;
    }

    template <class... Types>
    bool TMultiInput<Types...>::setInput(IParam* input)
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        bool success = false;
        if (m_current_input)
        {
            selectType<Types...>(*this, m_current_input->getTypeInfo(), static_cast<IParam*>(nullptr), &success);
            success = false;
        }
        selectType<Types...>(*this, input->getTypeInfo(), input, &success);
        if (success)
        {
            m_current_input = input;
        }
        return success;
    }

    template <class... Types>
    bool TMultiInput<Types...>::getInput(const OptionalTime_t& ts, size_t* fn)
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        bool success = false;
        if (m_current_input)
        {
            selectType<Types...>(*this, m_current_input->getTypeInfo(), ts, fn, &success);
            this->_ts = m_current_input->getTimestamp();
            this->_fn = m_current_input->getFrameNumber();
        }
        return success;
    }

    template <class... Types>
    bool TMultiInput<Types...>::getInput(size_t fn, OptionalTime_t* ts)
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        bool success = false;
        if (m_current_input)
        {
            selectType<Types...>(*this, m_current_input->getTypeInfo(), fn, ts, &success);
        }
        else
        {
        }
        return success;
    }

    template <class... Types>
    void TMultiInput<Types...>::setMtx(Mutex_t* mtx)
    {
        InputParam::setMtx(mtx);
        typeLoop<Types...>(*this, mtx);
    }

    template <class... Types>
    mo::TypeInfo TMultiInput<Types...>::getTypeInfo() const
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        if (m_current_input == nullptr)
        {
            return _void_type_info;
        }
        else
        {
            return m_current_input->getTypeInfo();
        }
    }

    template <class... Types>
    mo::IParam* TMultiInput<Types...>::getInputParam() const
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        return m_current_input;
    }

    template <class... Types>
    OptionalTime_t TMultiInput<Types...>::getInputTimestamp()
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        if (m_current_input)
        {
            return m_current_input->getTimestamp();
        }
        return {};
    }

    template <class... Types>
    size_t TMultiInput<Types...>::getInputFrameNumber()
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        if (m_current_input)
        {
            return m_current_input->getFrameNumber();
        }
        return std::numeric_limits<size_t>::max();
    }

    template <class... Types>
    bool TMultiInput<Types...>::isInputSet() const
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        return m_current_input != nullptr;
    }

    template <class... Types>
    bool TMultiInput<Types...>::acceptsInput(IParam* input) const
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        AcceptInputRedirect<TMultiInput<Types...>> redirect(*this);
        bool success = false;
        selectType<Types...>(redirect, input->getTypeInfo(), input, &success);
        return success;
    }

    template <class... Types>
    bool TMultiInput<Types...>::acceptsType(const TypeInfo& type) const
    {
        mo::Mutex_t::scoped_lock lock(mtx());
        AcceptInputRedirect<TMultiInput<Types...>> redirect(*this);
        bool success = false;
        selectType<Types...>(redirect, type, type, &success);
        return success;
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(std::shared_ptr<IParam> input, bool* success)
    {
        *success = mo::get<TInputParamPtr<T>>(m_inputs).setInput(input);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(IParam* input, bool* success)
    {
        *success = mo::get<TInputParamPtr<T>>(m_inputs).setInput(input);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(std::tuple<const Types*...>* user_var_)
    {
        mo::get<TInputParamPtr<T>>(m_inputs).setUserDataPtr(&mo::get<const T*>(*user_var_));
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(Mutex_t* mtx)
    {
        mo::get<TInputParamPtr<T>>(m_inputs).setMtx(mtx);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(Context* ctx)
    {
        mo::get<TInputParamPtr<T>>(m_inputs).setContext(ctx);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(const OptionalTime_t& ts, size_t* fn, bool* success)
    {
        *success = mo::get<TInputParamPtr<T>>(m_inputs).getInput(ts, fn);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(size_t fn, OptionalTime_t* ts, bool* success)
    {
        *success = mo::get<TInputParamPtr<T>>(m_inputs).getInput(fn, ts);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::acceptsInput(IParam* input, bool* success) const
    {
        *success = mo::get<TInputParamPtr<T>>(m_inputs).acceptsInput(input);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::acceptsInput(const TypeInfo& type, bool* success) const
    {
        *success = mo::get<TInputParamPtr<T>>(m_inputs).acceptsType(type);
    }

    template <class... Types>
    template<class T, class Slot>
    void TMultiInput<Types...>::apply(std::vector<std::shared_ptr<Connection>>& connections, Slot slot)
    {
        connections.push_back(mo::get<TInputParamPtr<T>>(m_inputs).registerUpdateNotifier(slot));
    }

    template <class... Types>
    std::shared_ptr<Connection> TMultiInput<Types...>::registerUpdateNotifier(UpdateSlot_t* f)
    {
        std::vector<std::shared_ptr<Connection>> out_connection;
        typeLoop<Types...>(*this, out_connection, f);
        return std::make_shared<MultiConnection>(std::move(out_connection));
    }

    template <class... Types>
    std::shared_ptr<Connection> TMultiInput<Types...>::registerUpdateNotifier(std::shared_ptr<TSignalRelay<UpdateSig_t>>& relay)
    {
        std::vector<std::shared_ptr<Connection>> out_connection;
        typeLoop<Types...>(*this, out_connection, relay);
        return std::make_shared<MultiConnection>(std::move(out_connection));
    }

    template <class... Types>
    std::shared_ptr<Connection> TMultiInput<Types...>::registerUpdateNotifier(ISlot* f)
    {
        std::vector<std::shared_ptr<Connection>> out_connection;
        typeLoop<Types...>(*this, out_connection, f);
        return std::make_shared<MultiConnection>(std::move(out_connection));
    }

    template <class... Types>
    std::shared_ptr<Connection> TMultiInput<Types...>::registerUpdateNotifier(std::shared_ptr<ISignalRelay> relay)
    {
        std::vector<std::shared_ptr<Connection>> out_connection;
        typeLoop<Types...>(*this, out_connection, relay);
        return std::make_shared<MultiConnection>(std::move(out_connection));
    }

    template <class... Types>
    template<class T>
    void TMultiInput<Types...>::apply(bool* modified) const
    {
        const auto val = mo::get<TInputParamPtr<T>>(m_inputs).modified();
        if(val)
        {
            *modified = val;
        }
    }

    template <class... Types>
    template<class T>
    void TMultiInput<Types...>::apply(bool modified, const ModifiedTag)
    {
        mo::get<TInputParamPtr<T>>(m_inputs).modified(modified);
    }

    template <class... Types>
    bool TMultiInput<Types...>::modified() const
    {
        if(m_current_input)
        {
            return m_current_input->modified();
        }
        return false;
    }

    template <class... Types>
    void TMultiInput<Types...>::modified(bool value)
    {
        typeLoop<Types...>(*this, value, ModifiedTag{});
    }

} // namespace mo
#endif // TMULTIINPUTINL_HPP
