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
#pragma once
#include "ITAccessibleParam.hpp"
#include "ITInputParam.hpp"
#include "MetaParam.hpp"
#include "ParamConstructor.hpp"

namespace mo
{

    // Meant to reference a pointer variable in user space, and to update that variable whenever
    // IE int* myVar;
    // auto TParam = TInputParamPtr(&myVar); // TInputParam now updates myvar to point to whatever the
    // input variable is for TParam.
    template <typename T>
    class MO_EXPORTS TInputParamPtr : virtual public ITInputParam<T>
    {
      public:
        using Input_t = const T*;
        using TContainerPtr_t = typename TParam<T>::TContainerPtr_t;
        using IContainerPtr_t = typename ITInputParam<T>::IContainerPtr_t;
        using IContainerConstPtr_t = typename ITInputParam<T>::IContainerConstPtr_t;

        TInputParamPtr(const std::string& name = "", const T** user_var_ = nullptr)
            : ITInputParam<T>(name)
            , TParam<T>(name, ParamFlags::kINPUT)
            , IParam(name, ParamFlags::kINPUT)
            , m_user_var(user_var_)
        {
        }

        void setUserDataPtr(const T** user_var_)
        {
            Lock_t lock(this->mtx());
            m_user_var = user_var_;
        }

        IContainerPtr_t getData(const Header& desired = Header())
        {
            auto data = ITInputParam<T>::getData(desired);
            if (!data)
            {
                return data;
            }
            if (data->getType() == ITInputParam<T>::getTypeInfo())
            {
                auto typed = std::static_pointer_cast<TDataContainer<T>>(data);
                if (typed && m_user_var)
                {
                    *m_user_var = typed->ptr();
                }
            }
            return data;
        }

        IContainerConstPtr_t getData(const Header& desired = Header()) const
        {
            return ITInputParam<T>::getData(desired);
        }

      protected:
        void updateDataImpl(const TContainerPtr_t& data, mo::UpdateFlags fg = UpdateFlags::kINPUT_UPDATED) override
        {
            Lock_t lock(this->mtx());
            ITInputParam<T>::updateDataImpl(data, fg);
            if (m_user_var)
            {
                *m_user_var = &data->data;
            }
        }

        const T** m_user_var; // Pointer to the user space pointer variable of type T
    };

    // Deprecated in favor of ct::TArrayView
    template <typename T, class A>
    class MO_DEPRECATED TInputParamPtr<std::vector<T, A>> : virtual public ITInputParam<std::vector<T, A>>
    {
      public:
        using Input_t = const std::vector<T, A>*;
        using TContainerPtr_t = typename TParam<std::vector<T, A>>::TContainerPtr_t;
        using IContainerPtr_t = typename ITInputParam<std::vector<T, A>>::IContainerPtr_t;
        using IContainerConstPtr_t = typename ITInputParam<std::vector<T, A>>::IContainerConstPtr_t;

        TInputParamPtr(const std::string& name = "", const std::vector<T, A>** user_var_ = nullptr)
            : ITInputParam<std::vector<T, A>>(name)
            , TParam<std::vector<T, A>>(name, ParamFlags::kINPUT)
            , IParam(name, ParamFlags::kINPUT)
            , m_user_var(user_var_)
        {
        }

        void setUserDataPtr(const std::vector<T, A>* user_var_)
        {
            Lock_t lock(this->mtx());
            m_user_var = user_var_;
        }

        IContainerPtr_t getData(const Header& desired = Header())
        {
            auto data = ITInputParam<std::vector<T, A>>::getData(desired);
            if (data->getType() == ITInputParam<std::vector<T, A>>::getTypeInfo())
            {
                auto typed = std::static_pointer_cast<TDataContainer<std::vector<T, A>>>(data);
                if (typed && m_user_var)
                {
                    *m_user_var = typed->ptr();
                }
            }
            return data;
        }

        IContainerConstPtr_t getData(const Header& desired = Header()) const
        {
            return ITInputParam<T>::getData(desired);
        }

      protected:
        void updateDataImpl(const TContainerPtr_t& data, mo::UpdateFlags fg = UpdateFlags::kINPUT_UPDATED) override
        {
            Lock_t lock(this->mtx());
            ITInputParam<std::vector<T, A>>::updateDataImpl(data, fg);
            if (m_user_var)
            {
                *m_user_var = &data->data;
            }
        }

        const std::vector<T, A>* m_user_var; // Pointer to the user space pointer variable of type T
    };

    template <typename T>
    class MO_EXPORTS TInputParamPtr<ct::TArrayView<T>> : virtual public ITInputParam<ct::TArrayView<T>>
    {
      public:
        using Input_t = ct::TArrayView<const T>;
        using TContainerPtr_t = typename TParam<ct::TArrayView<T>>::TContainerPtr_t;
        using IContainerPtr_t = typename ITInputParam<ct::TArrayView<T>>::IContainerPtr_t;
        using IContainerConstPtr_t = typename ITInputParam<ct::TArrayView<T>>::IContainerConstPtr_t;

        TInputParamPtr(const std::string& name = "", ct::TArrayView<const T>* user_var_ = nullptr)
            : ITInputParam<ct::TArrayView<T>>(name)
            , TParam<ct::TArrayView<T>>(name, ParamFlags::kINPUT)
            , IParam(name, ParamFlags::kINPUT)
            , m_user_var(user_var_)
        {
        }

        void setUserDataPtr(ct::TArrayView<const T>* user_var_)
        {
            Lock_t lock(this->mtx());
            m_user_var = user_var_;
        }

        IContainerPtr_t getData(const Header& desired = Header())
        {
            auto data = ITInputParam<ct::TArrayView<T>>::getData(desired);
            if (data->getType() == ITInputParam<ct::TArrayView<T>>::getTypeInfo())
            {
                auto typed = std::static_pointer_cast<TDataContainer<ct::TArrayView<T>>>(data);
                if (typed && m_user_var)
                {
                    *m_user_var = typed->data;
                }
            }
            return data;
        }

        IContainerConstPtr_t getData(const Header& desired = Header()) const
        {
            return ITInputParam<ct::TArrayView<T>>::getData(desired);
        }

      protected:
        void updateDataImpl(const TContainerPtr_t& data, mo::UpdateFlags fg = UpdateFlags::kINPUT_UPDATED) override
        {
            Lock_t lock(this->mtx());
            ITInputParam<ct::TArrayView<T>>::updateDataImpl(data, fg);
            if (m_user_var)
            {
                *m_user_var = data->data;
            }
        }

        ct::TArrayView<const T>* m_user_var; // Pointer to the user space pointer variable of type T
    };

    template <typename T>
    class MO_EXPORTS TInputParamPtr<std::shared_ptr<T>> : virtual public ITInputParam<T>
    {
      public:
        using Input_t = std::shared_ptr<const T>;
        using TContainerPtr_t = typename TParam<T>::TContainerPtr_t;
        using IContainerPtr_t = typename ITInputParam<T>::IContainerPtr_t;
        using IContainerConstPtr_t = typename ITInputParam<T>::IContainerConstPtr_t;

        TInputParamPtr(const std::string& name = "", std::shared_ptr<const T>* user_var_ = nullptr)
            : ITInputParam<T>(name)
            , m_user_var(user_var_)
        {
        }

        void setUserDataPtr(std::shared_ptr<const T>* user_var_)
        {
            Lock_t lock(this->mtx());
            m_user_var = user_var_;
        }

        virtual void updateDataImpl(const TContainerPtr_t& data)
        {
            Lock_t lock(this->mtx());
            ITInputParam<T>::updateDataImpl(data);
            if (m_user_var)
            {
                *m_user_var = data;
            }
        }

        IContainerPtr_t getData(const Header& desired = Header())
        {
            auto data = ITInputParam<T>::getData(desired);
            auto typed = std::static_pointer_cast<TDataContainer<T>>(data);
            if (typed && m_user_var)
            {
                *m_user_var = typed->sharedPtr();
            }
        }

        IContainerConstPtr_t getData(const Header& desired = Header()) const
        {
            return ITInputParam<T>::getData(desired);
        }

      protected:
        std::shared_ptr<const T>* m_user_var; // Pointer to the user space pointer variable of type T
    };
}
