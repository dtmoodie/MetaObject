#pragma once
#include "TInputParam.hpp"
#include "TypeSelector.hpp"

namespace mo
{
    template <class T, class U>
    constexpr int indexOfHelper(int idx, std::tuple<U>* = nullptr)
    {
        return idx;
    }
    template <class T, class U, class... Ts>
    constexpr int indexOfHelper(int idx, std::tuple<U, Ts...>* = nullptr)
    {
        return std::is_same<T, U>::value ? idx
                                         : indexOfHelper<T, Ts...>(idx - 1, static_cast<std::tuple<Ts...>*>(nullptr));
    }

    template <class T, class... Ts>
    constexpr int indexOf()
    {
        return sizeof...(Ts)-indexOfHelper<T, Ts...>(sizeof...(Ts), static_cast<std::tuple<Ts...>*>(nullptr));
    }

    template <class T, class... Ts>
    T& get(std::tuple<Ts...>& tuple)
    {
        return std::get<indexOf<T, Ts...>()>(tuple);
    }

    template <class T, class... Ts>
    const T& get(const std::tuple<Ts...>& tuple)
    {
        static const int IDX = indexOf<T, Ts...>();
        return std::get<IDX>(tuple);
    }

    template <class T>
    struct AcceptInputRedirect
    {
        AcceptInputRedirect(const T& func) : m_func(func) {}
        template <class Type, class... Args>
        void apply(Args&&... args)
        {
            m_func.template acceptsInput<Type>(std::forward<Args>(args)...);
        }
        const T& m_func;
    };

    template <class... Types>
    class TMultiInput : public InputParam
    {
      public:
        void setUserDataPtr(std::tuple<const Types*...>* user_var_) { typeLoop<Types...>(*this, user_var_); }

        virtual bool setInput(std::shared_ptr<IParam> input) override
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

        virtual bool setInput(IParam* input) override
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

        virtual bool getInput(const OptionalTime_t& ts, size_t* fn = nullptr) override
        {
            bool success = false;
            if (m_current_input)
            {
                selectType<Types...>(*this, m_current_input->getTypeInfo(), ts, fn, &success);
            }
            return success;
        }

        virtual bool getInput(size_t fn, OptionalTime_t* ts = nullptr) override
        {
            bool success = false;
            if (m_current_input)
            {
                selectType<Types...>(*this, m_current_input->getTypeInfo(), fn, ts, &success);
            }
            return success;
        }

        virtual void setMtx(Mutex_t* mtx) override { typeLoop<Types...>(*this, mtx); }

        const mo::TypeInfo& getTypeInfo() const
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

        mo::IParam* getInputParam() const { return m_current_input; }

        OptionalTime_t getInputTimestamp()
        {
            if (m_current_input)
                return m_current_input->getTimestamp();
            return {};
        }

        size_t getInputFrameNumber()
        {
            if (m_current_input)
            {
                return m_current_input->getFrameNumber();
            }
            return std::numeric_limits<size_t>::max();
        }

        bool isInputSet() const { return m_current_input != nullptr; }

        bool acceptsInput(IParam* input) const
        {
            AcceptInputRedirect<TMultiInput<Types...>> redirect(*this);
            bool success = false;
            selectType<Types...>(redirect, input->getTypeInfo(), input, &success);
            return success;
        }

        bool acceptsType(const TypeInfo& type) const
        {
            AcceptInputRedirect<TMultiInput<Types...>> redirect(*this);
            bool success = false;
            selectType<Types...>(redirect, type, type, &success);
            return success;
        }

        template <class T>
        void apply(std::shared_ptr<IParam> input, bool* success)
        {
            *success = get<TInputParamPtr<T>>(m_inputs).setInput(input);
        }

        template <class T>
        void apply(IParam* input, bool* success)
        {
            *success = get<TInputParamPtr<T>>(m_inputs).setInput(input);
        }

        template <class T>
        void apply(std::tuple<const Types*...>* user_var_)
        {
            get<TInputParamPtr<T>>(m_inputs).setUserDataPtr(&get<const T*>(*user_var_));
        }

        template <class T>
        void apply(Mutex_t* mtx)
        {
            get<TInputParamPtr<T>>(m_inputs).setMtx(mtx);
        }

        template <class T>
        void apply(Context* ctx)
        {
            get<TInputParamPtr<T>>(m_inputs).setContext(ctx);
        }

        template <class T>
        void apply(const OptionalTime_t& ts, size_t* fn, bool* success)
        {
            *success = get<TInputParamPtr<T>>(m_inputs).getInput(ts, fn);
        }

        template <class T>
        void apply(size_t fn, OptionalTime_t* ts, bool* success)
        {
            *success = get<TInputParamPtr<T>>(m_inputs).getInput(fn, ts);
        }

        template <class T>
        void acceptsInput(IParam* input, bool* success) const
        {
            *success = get<TInputParamPtr<T>>(m_inputs).acceptsInput(input);
        }

        template <class T>
        void acceptsInput(const TypeInfo& type, bool* success) const
        {
            *success = get<TInputParamPtr<T>>(m_inputs).acceptsType(type);
        }

      private:
        static mo::TypeInfo _void_type_info;
        std::tuple<TInputParamPtr<Types>...> m_inputs;
        mo::IParam* m_current_input = nullptr;
    };

    template <class... T>
    mo::TypeInfo TMultiInput<T...>::_void_type_info = mo::TypeInfo(typeid(void));
}
