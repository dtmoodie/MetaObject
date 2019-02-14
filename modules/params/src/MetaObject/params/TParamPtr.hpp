#pragma once
#include "MetaParam.hpp"
#include "OutputParam.hpp"
#include "TParam.hpp"
#include "detail/print_data.hpp"

namespace mo
{
    /*! The TParamPtr class is a concrete implementation of TParam
     *  which implements wrapping of a raw pointer to user data.  This is used
     *  extensively inside of the PARAM macro as follows:
     *
     *  float user_data;
     *  TParamPtr<float> user_param("float_data", &user_data);
     *  user_param.updateData(10);
     *  user_data == 10;
     *
     *  This code snipit creates a user space variable 'user_data'
     *  which is wrapped for reflection purposes by 'user_param' named 'float_data'.
     *  Updates to user_param are reflected in user_data
     */
    template <typename T>
    struct MO_EXPORTS TParamPtr : public TParam<T>
    {
        using TContainerPtr_t = typename TParam<T>::TContainerPtr_t;

        TParamPtr(const std::string& name = "",
                  T* ptr = nullptr,
                  ParamFlags type = ParamFlags::Control_e,
                  bool owns_data = false)
            : TParam<T>(name, type)
            , m_ptr(ptr)
        {
            if (owns_data)
            {
                m_owner.reset(ptr);
            }
        }

        ~TParamPtr() = default;

        void updatePtr(T* ptr, bool owns_data = false)
        {
            Lock_t lock(this->mtx());
            m_owner.reset();
            if (owns_data)
            {
                m_owner.reset(ptr);
            }
            m_ptr = ptr;
        }

        IParam* emitUpdate(const Header& header, UpdateFlags = ValueUpdated_e) override
        {
            Lock_t lock(this->mtx());
            if (m_ptr)
            {
                TParam<T>::updateData(*m_ptr, header);
            }
            return this;
        }

        // commit a Param's value copying metadata info from another parmaeter
        IParam* emitUpdate(const IParam& other, UpdateFlags = ValueUpdated_e) override
        {
            Lock_t lock(this->mtx());
            if (m_ptr)
            {
                TParam<T>::updateData(*m_ptr, mo::tag::_param = other);
            }
            return this;
        }

        std::ostream& print(std::ostream& os) const override
        {
            Lock_t lock(this->mtx());
            TParam<T>::print(os);
            os << ' ';
            if (m_ptr)
            {
                mo::print(os, *m_ptr);
            }
            return os;
        }

        bool isValid() const override
        {
            return TParam<T>::isValid() || m_ptr != nullptr;
        }

        ConstAccessToken<T> read() const
        {
            mo::Lock_t lock(this->mtx());
            if (TParam<T>::isValid())
            {
                return TParam<T>::read();
            }

            MO_ASSERT(m_ptr != nullptr);
            return {std::move(lock), *this, *m_ptr};
        }

        AccessToken<T> access()
        {
            mo::Lock_t lock(this->mtx());
            if (TParam<T>::isValid())
            {
                return TParam<T>::access();
            }

            MO_ASSERT(m_ptr != nullptr);
            return {std::move(lock), *this, *m_ptr};
        }

      protected:
        void updateDataImpl(const TContainerPtr_t& data)
        {
            TParam<T>::updateDataImpl(data);
            updateUserData(data->data);
        }

        T* ptr()
        {
            return m_ptr;
        }

        void updateUserData(const T& data)
        {
            if (&data != m_ptr)
            {
                if (m_ptr)
                {
                    *m_ptr = data;
                }
            }
        }

      private:
        T* m_ptr;
        std::unique_ptr<T> m_owner;
    };

    template <typename T>
    struct MO_EXPORTS TParamPtr<std::shared_ptr<T>> : virtual public TParam<T>
    {
        using TContainerPtr_t = typename TParam<T>::TContainerPtr_t;

        TParamPtr(const std::string& name = "",
                  std::shared_ptr<T>* ptr = nullptr,
                  ParamFlags type = ParamFlags::Control_e,
                  bool owns_data = false)
            : TParam<T>(name, type)
            , m_ptr(ptr)
        {
            if (owns_data)
            {
                m_owner.reset(ptr);
            }
        }

        void updatePtr(std::shared_ptr<T>* ptr, bool owns_data = false)
        {
            Lock_t lock(this->mtx());
            m_owner.reset();
            if (owns_data)
            {
                m_owner.reset(ptr);
            }

            m_ptr = ptr;
        }

        void updateData(const TContainerPtr_t& data) override
        {
            TParam<T>::updateData(data);
            updateUserData(data);
        }

        std::ostream& print(std::ostream& os) const override
        {
            Lock_t lock(this->mtx());
            TParam<T>::print(os);
            os << ' ';
            if (m_ptr)
            {
                mo::print(os, *m_ptr);
            }
            return os;
        }

      protected:
        void updateUserData(const TContainerPtr_t& data)
        {
            TParam<T>::updateDataImpl(data);
            if (m_ptr)
            {
                *m_ptr = data;
            }
        }
        std::shared_ptr<T>* ptr()
        {
            return m_ptr;
        }

      private:
        std::shared_ptr<T>* m_ptr;
        std::unique_ptr<std::shared_ptr<T>> m_owner;
    };
}
