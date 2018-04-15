#pragma once

#include "TInputParam.hpp"
#include "TypeSelector.hpp"

namespace mo
{
    template <class... Types>
    class TMultiInput : public InputParam
    {
      public:
        void setUserDataPtr(std::tuple<Types*...>* user_var_) { typeLoop(*this, user_var_); }

        virtual bool setInput(std::shared_ptr<IParam> input) override
        {
            if (m_current_input)
            {
                selectType(*this, m_current_input->getTypeInfo(), static_cast<IParam*>(nullptr));
            }
            bool success = false;
            selectType(*this, input->getTypeInfo(), &success);
            if (success)
            {
                m_current_input = input.get();
            }
            return success;
        }

        virtual bool setInput(IParam* input) override
        {
            if (m_current_input)
            {
                selectType(*this, m_current_input->getTypeInfo(), static_cast<IParam*>(nullptr));
            }
            bool success = false;
            selectType(*this, input->getTypeInfo(), &success);
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
                selectType(*this, m_current_input->getTypeInfo(), ts, fn, &success);
            }
            return success;
        }

        virtual bool getInput(size_t fn, OptionalTime_t* ts = nullptr) override
        {
            bool success = false;
            if (m_current_input)
            {
                selectType(*this, m_current_input->getTypeInfo(), fn, ts, &success);
            }
            return success;
        }

        virtual void setMtx(Mutex_t* mtx) override { typeLoop<Types...>(*this, mtx); }

        template <class T>
        void apply(std::shared_ptr<IParam> input, bool* success)
        {
            *success = std::get<TInputParamPtr<T>>(m_inputs).setInput(input);
        }

        template <class T>
        void apply(IParam* input, bool* success)
        {
            *success = std::get<TInputParamPtr<T>>(m_inputs).setInput(input);
        }

        template <class T>
        void apply(std::tuple<Types*...>* user_var_)
        {
            std::get<TInputParamPtr<T>>(m_inputs).setUserDataPtr(&std::get<T*>(*user_var_));
        }

        template <class T>
        void apply(Mutex_t* mtx)
        {
            std::get<TInputParamPtr<T>>(m_inputs).setMtx(mtx);
        }

        template <class T>
        void apply(Context* ctx)
        {
            std::get<TInputParamPtr<T>>(m_inputs).setContext(ctx);
        }

        template <class T>
        void apply(const OptionalTime_t& ts, size_t* fn, bool* success)
        {
            *success = std::get<TInputParamPtr<T>>(m_inputs).getInput(ts, fn);
        }

        template <class T>
        void apply(size_t fn, OptionalTime_t* ts, bool* success)
        {
            *success = std::get<TInputParamPtr<T>>(m_inputs).getInput(fn, ts);
        }

      private:
        std::tuple<TInputParamPtr<Types>...> m_inputs;
        mo::IParam* m_current_input;
    };
}
