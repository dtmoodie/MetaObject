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
        bool success = false;
        if (m_current_input)
        {
            selectType<Types...>(*this, m_current_input->getTypeInfo(), static_cast<IParam*>(nullptr), &success);
            success = false;
        }
        selectType<Types...>(*this, input->getTypeInfo(), input.get(), &success);
        if (success)
        {
            m_current_input = input.get();
        }
        return success;
    }

    template <class... Types>
    bool TMultiInput<Types...>::setInput(IParam* input)
    {
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
        bool success = false;
        if (m_current_input)
        {
            selectType<Types...>(*this, m_current_input->getTypeInfo(), ts, fn, &success);
        }
        return success;
    }

    template <class... Types>
    bool TMultiInput<Types...>::getInput(size_t fn, OptionalTime_t* ts)
    {
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
        typeLoop<Types...>(*this, mtx);
    }

    template <class... Types>
    mo::TypeInfo TMultiInput<Types...>::getTypeInfo() const
    {
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
        return m_current_input;
    }

    template <class... Types>
    OptionalTime_t TMultiInput<Types...>::getInputTimestamp()
    {
        if (m_current_input)
            return m_current_input->getTimestamp();
        return {};
    }

    template <class... Types>
    size_t TMultiInput<Types...>::getInputFrameNumber()
    {
        if (m_current_input)
        {
            return m_current_input->getFrameNumber();
        }
        return std::numeric_limits<size_t>::max();
    }

    template <class... Types>
    bool TMultiInput<Types...>::isInputSet() const
    {
        return m_current_input != nullptr;
    }

    template <class... Types>
    bool TMultiInput<Types...>::acceptsInput(IParam* input) const
    {
        AcceptInputRedirect<TMultiInput<Types...>> redirect(*this);
        bool success = false;
        selectType<Types...>(redirect, input->getTypeInfo(), input, &success);
        return success;
    }

    template <class... Types>
    bool TMultiInput<Types...>::acceptsType(const TypeInfo& type) const
    {
        AcceptInputRedirect<TMultiInput<Types...>> redirect(*this);
        bool success = false;
        selectType<Types...>(redirect, type, type, &success);
        return success;
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(std::shared_ptr<IParam> input, bool* success)
    {
        *success = get<TInputParamPtr<T>>(m_inputs).setInput(input);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(IParam* input, bool* success)
    {
        *success = get<TInputParamPtr<T>>(m_inputs).setInput(input);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(std::tuple<const Types*...>* user_var_)
    {
        get<TInputParamPtr<T>>(m_inputs).setUserDataPtr(&get<const T*>(*user_var_));
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(Mutex_t* mtx)
    {
        get<TInputParamPtr<T>>(m_inputs).setMtx(mtx);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(Context* ctx)
    {
        get<TInputParamPtr<T>>(m_inputs).setContext(ctx);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(const OptionalTime_t& ts, size_t* fn, bool* success)
    {
        *success = get<TInputParamPtr<T>>(m_inputs).getInput(ts, fn);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::apply(size_t fn, OptionalTime_t* ts, bool* success)
    {
        *success = get<TInputParamPtr<T>>(m_inputs).getInput(fn, ts);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::acceptsInput(IParam* input, bool* success) const
    {
        *success = get<TInputParamPtr<T>>(m_inputs).acceptsInput(input);
    }

    template <class... Types>
    template <class T>
    void TMultiInput<Types...>::acceptsInput(const TypeInfo& type, bool* success) const
    {
        *success = get<TInputParamPtr<T>>(m_inputs).acceptsType(type);
    }

} // namespace mo
#endif // TMULTIINPUTINL_HPP
