#pragma once
#include "ITAccessibleParam.hpp"
#include "MetaParam.hpp"
#include "OutputParam.hpp"
#include "detail/print_data.hpp"

namespace mo
{
    /*! The TParamPtr class is a concrete implementation of ITParam
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
    struct MO_EXPORTS TParamPtr : virtual public ITParam<T>
    {
        using TContainerPtr_t = typename ITParam<T>::TContainerPtr_t;

        TParamPtr(const std::string& name = "",
                  T* ptr = nullptr,
                  ParamFlags type = ParamFlags::Control_e,
                  bool owns_data = false)
            : ITParam<T>(name, type)
            , m_ptr(ptr)
            , m_owns_data(owns_data)
        {
        }

        ~TParamPtr()
        {
            if (m_owns_data)
            {
                delete m_ptr;
            }
        }

        void updatePtr(T* ptr, bool owns_data = false)
        {
            Lock lock(this->mtx());
            if (m_ptr && m_owns_data)
            {
                delete m_ptr;
            }
            m_ptr = ptr;
            m_owns_data = owns_data;
        }
        void updateData(const TContainerPtr_t& data)
        {
            ITParam<T>::updateData(data);
            updateUserData(*data);
        }

        virtual IParam* emitUpdate(const Header& header, UpdateFlags = ValueUpdated_e) override
        {
            Lock lock(this->mtx());
            if (m_ptr)
            {
                ITParam<T>::updateData(*m_ptr, header);
            }
        }

        // commit a Param's value copying metadata info from another parmaeter
        virtual IParam* emitUpdate(const IParam& other, UpdateFlags flags_ = ValueUpdated_e) override
        {
            Lock lock(this->mtx());
            if (m_ptr)
            {
                ITParam<T>::updateData(*m_ptr, mo::tag::_param = other);
            }
        }

        std::ostream& print(std::ostream& os) const override
        {
            Lock lock(this->mtx());
            ITParam<T>::print(os);
            os << ' ';
            if (m_ptr)
            {
                mo::print(os, *m_ptr);
            }
            return os;
        }

      protected:
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
        bool m_owns_data;
    };

    template <typename T>
    struct MO_EXPORTS TParamPtr<std::shared_ptr<T>> : virtual public ITParam<T>
    {
        using TContainerPtr_t = typename ITParam<T>::TContainerPtr_t;

        TParamPtr(const std::string& name = "",
                  std::shared_ptr<T>* ptr = nullptr,
                  ParamFlags type = ParamFlags::Control_e,
                  bool owns_data = false)
            : ITParam<T>(name, type)
            , m_ptr(ptr)
            , m_owns_data(owns_data)
        {
        }

        ~TParamPtr()
        {
            if (m_owns_data)
            {
                delete m_ptr;
            }
        }

        void updatePtr(std::shared_ptr<T>* ptr, bool owns_data = false)
        {
            Lock lock(this->mtx());
            if (m_ptr && m_owns_data)
            {
                delete m_ptr;
            }
            m_ptr = ptr;
            m_owns_data = owns_data;
        }

        virtual void updateData(const TContainerPtr_t& data) override
        {
            ITParam<T>::updateData(data);
            updateUserData(data);
        }

        std::ostream& print(std::ostream& os) const override
        {
            Lock lock(this->mtx());
            ITParam<T>::print(os);
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
        bool m_owns_data;
    };

    /*!
     * TParamOutput is used with the OUTPUT macro.  In this case, the param owns the data and the owning parent object
     * owns a reference to the data which is updated by the param's reset function.
     */
    template <typename T>
    struct MO_EXPORTS TParamOutput : virtual public TParamPtr<T>, virtual public OutputParam
    {
        TParamOutput()
            : IParam(mo::tag::_param_flags = mo::ParamFlags::Output_e)
        {
        }

        virtual std::vector<TypeInfo> listOutputTypes() const override;
        ParamBase* getOutputParam(const TypeInfo type) override;
        const ParamBase* getOutputParam(const TypeInfo type) const override;
        ParamBase* getOutputParam() override;
        const ParamBase* getOutputParam() const override;
    };
}
