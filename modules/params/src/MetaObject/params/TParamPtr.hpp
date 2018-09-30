#pragma once
#include "ITAccessibleParam.hpp"
#include "MetaParam.hpp"
#include "OutputParam.hpp"

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
        TParamPtr(const std::string& name = "",
                  T* ptr = nullptr,
                  ParamFlags type = ParamFlags::Control_e,
                  bool owns_data = false)
            : ITParam<T>(name, type) m_ptr(ptr), m_owns_data(owns_data)
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
            mo::Mutex_t::scoped_lock lock(this->mtx());
            if (m_ptr && m_owns_data)
            {
                delete m_ptr;
            }
            m_ptr = ptr;
            m_owns_data = owns_data;
        }

        template <class... Args>
        void updateData(const T& data, const Args&... args)
        {
            mo::Mutex_t::scoped_lock lock(this->mtx());
            ITParam<T>::updateData(data, std::forward<Args>(args)...);
            updateUserData();
        }

        template <class... Args>
        void updateData(T&& data, const Args&... args)
        {
            mo::Mutex_t::scoped_lock lock(this->mtx());
            ITParam<T>::updateData(data, std::forward<Args>(args)...);
            updateUserData();
        }

        virtual void updateData(const T& data, const Header& header) override
        {
            mo::Mutex_t::scoped_lock lock(this->mtx());
            ITParam<T>::updateData(data, header);
            updateUserData();
        }

        virtual void updateData(T&& data, Header&& header) override
        {
            mo::Mutex_t::scoped_lock lock(this->mtx());
            ITParam<T>::updateData(std::move(data), std::move(header));
            updateUserData();
        }

        virtual IParam* emitUpdate(const Header& header = Header(), UpdateFlags) override
        {
            mo::Mutex_t::scoped_lock lock(this->mtx());
            if (m_ptr)
            {
                ITParam<T>::updateData(*m_ptr, header);
            }
        }

        // commit a Param's value copying metadata info from another parmaeter
        virtual IParam* emitUpdate(const IParam& other) override
        {
            mo::Mutex_t::scoped_lock lock(this->mtx());
            if (m_ptr)
            {
                ITParam<T>::updateData(*m_ptr, mo::tag::_param = other);
            }
        }

        std::ostream& print(std::ostream& os) const override
        {
            mo::Mutex_t::scoped_lock lock(this->mtx());
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
        TParamPtr(const std::string& name = "",
                  std::shared_ptr<T>* ptr = nullptr,
                  ParamFlags type = ParamFlags::Control_e,
                  bool owns_data = false)
            : ITParam<T>(name, type) m_ptr(ptr), m_owns_data(owns_data)
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
            if (m_ptr && m_owns_data)
            {
                delete m_ptr;
            }
            m_ptr = ptr;
            m_owns_data = owns_data;
        }

        template <class... Args>
        void updateData(const T& data, const Args&... args)
        {
            ITParam<T>::updateData(data, std::forward<Args>(args)...);
            updateUserData();
        }

        template <class... Args>
        void updateData(T&& data, const Args&... args)
        {
            ITParam<T>::updateData(data, std::forward<Args>(args)...);
            updateUserData();
        }

        virtual void updateData(const T& data, const Header& header) override
        {
            ITParam<T>::updateData(data, header);
            updateUserData();
        }

        virtual void updateData(T&& data, Header&& header) override
        {
            ITParam<T>::updateData(std::move(data), std::move(header));
            updateUserData();
        }

        std::ostream& print(std::ostream& os) const override
        {
            mo::Mutex_t::scoped_lock lock(this->mtx());
            ITParam<T>::print(os);
            os << ' ';
            if (m_ptr)
            {
                mo::print(os, *m_ptr);
            }
            return os;
        }

      protected:
        void updateUserData()
        {
            if (m_ptr)
            {
                ContainerPtr_t container;
                if (getData(container))
                {
                    *m_ptr = container;
                }
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
        TParamOutput() : IParam(mo::tag::_param_flags = mo::ParamFlags::Output_e)
        {
        }

        virtual std::vector<TypeInfo> listOutputTypes() const override;
        ParamBase* getOutputParam(const TypeInfo type) override;
        const ParamBase* getOutputParam(const TypeInfo type) const override;
        ParamBase* getOutputParam() override;
        const ParamBase* getOutputParam() const override;
    };
}
#include "detail/TParamPtrImpl.hpp"
