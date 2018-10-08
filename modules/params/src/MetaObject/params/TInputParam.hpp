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
        using TContainerPtr_t = typename ITParam<T>::TContainerPtr_t;

        TInputParamPtr(const std::string& name = "", T** user_var_ = nullptr)
            : ITInputParam<T>(name)
            , m_user_var(user_var_)
        {
        }

        void setUserDataPtr(T** user_var_)
        {
            Lock lock(this->mtx());
            m_user_var = user_var_;
        }

        virtual void updateData(const TContainerPtr_t& data)
        {
            Lock lock(this->mtx());
            ITInputParam<T>::updateData(data);
            if (m_user_var)
            {
                *m_user_var = data;
            }
        }

      protected:
        T** m_user_var; // Pointer to the user space pointer variable of type T
    };

    template <typename T>
    class MO_EXPORTS TInputParamPtr<std::shared_ptr<T>> : virtual public ITInputParam<T>
    {
      public:
        using TContainerPtr_t = typename ITParam<T>::TContainerPtr_t;

        TInputParamPtr(const std::string& name = "", std::shared_ptr<T>* user_var_ = nullptr)
            : ITInputParam<T>(name)
            , m_user_var(user_var_)
        {
        }

        void setUserDataPtr(std::shared_ptr<T>* user_var_)
        {
            Lock lock(this->mtx());
            m_user_var = user_var_;
        }

        virtual void updateData(const TContainerPtr_t& data)
        {
            Lock lock(this->mtx());
            ITInputParam<T>::updateData(data);
            if (m_user_var)
            {
                *m_user_var = data;
            }
        }

      protected:
        std::shared_ptr<T>* m_user_var; // Pointer to the user space pointer variable of type T
    };
}
