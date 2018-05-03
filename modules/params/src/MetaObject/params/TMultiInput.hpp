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
    inline T& get(std::tuple<Ts...>& tuple)
    {
        return std::get<indexOf<T, Ts...>()>(tuple);
    }

    template <class T, class... Ts>
    inline const T& get(const std::tuple<Ts...>& tuple)
    {
        return std::get<indexOf<T, Ts...>()>(tuple);
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

        virtual const mo::TypeInfo& getTypeInfo() const override
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

        virtual mo::IParam* getInputParam() const override { return m_current_input; }

        virtual OptionalTime_t getInputTimestamp() override
        {
            if (m_current_input)
                return m_current_input->getTimestamp();
            return {};
        }

        virtual size_t getInputFrameNumber() override
        {
            if (m_current_input)
            {
                return m_current_input->getFrameNumber();
            }
            return std::numeric_limits<size_t>::max();
        }

        virtual bool isInputSet() const override { return m_current_input != nullptr; }

        virtual bool acceptsInput(IParam* input) const override
        {
            AcceptInputRedirect<TMultiInput<Types...>> redirect(*this);
            bool success = false;
            selectType<Types...>(redirect, input->getTypeInfo(), input, &success);
            return success;
        }

        virtual bool acceptsType(const TypeInfo& type) const override
        {
            AcceptInputRedirect<TMultiInput<Types...>> redirect(*this);
            bool success = false;
            selectType<Types...>(redirect, type, type, &success);
            return success;
        }

        template <class T>
        inline void apply(std::shared_ptr<IParam> input, bool* success)
        {
            *success = get<TInputParamPtr<T>>(m_inputs).setInput(input);
        }

        template <class T>
        inline void apply(IParam* input, bool* success)
        {
            *success = get<TInputParamPtr<T>>(m_inputs).setInput(input);
        }

        template <class T>
        inline void apply(std::tuple<const Types*...>* user_var_)
        {
            get<TInputParamPtr<T>>(m_inputs).setUserDataPtr(&get<const T*>(*user_var_));
        }

        template <class T>
        inline void apply(Mutex_t* mtx)
        {
            get<TInputParamPtr<T>>(m_inputs).setMtx(mtx);
        }

        template <class T>
        inline void apply(Context* ctx)
        {
            get<TInputParamPtr<T>>(m_inputs).setContext(ctx);
        }

        template <class T>
        inline void apply(const OptionalTime_t& ts, size_t* fn, bool* success)
        {
            *success = get<TInputParamPtr<T>>(m_inputs).getInput(ts, fn);
        }

        template <class T>
        inline void apply(size_t fn, OptionalTime_t* ts, bool* success)
        {
            *success = get<TInputParamPtr<T>>(m_inputs).getInput(fn, ts);
        }

        template <class T>
        inline void acceptsInput(IParam* input, bool* success) const
        {
            *success = get<TInputParamPtr<T>>(m_inputs).acceptsInput(input);
        }

        template <class T>
        inline void acceptsInput(const TypeInfo& type, bool* success) const
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
