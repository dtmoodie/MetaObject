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
#include "MetaParam.hpp"
#include "ParamConstructor.hpp"
#include "TSubscriber.hpp"

namespace mo
{

    // Meant to reference a pointer variable in user space, and to update that variable whenever
    // IE int* myVar;
    // auto TParam = TSubscriberPtr(&myVar); // TInputParam now updates myvar to point to whatever the
    // input variable is for TParam.
    template <typename T>
    class MO_EXPORTS TSubscriberPtr : public TSubscriber<T>
    {
      public:
        using Input_t = const T*;

        TSubscriberPtr(const std::string& name = "",
                       const T** user_var_ = nullptr,
                       mo::ParamFlags flags = mo::ParamFlags::kINPUT)
            : m_user_var(user_var_)
        {
            this->setName(name);
            this->appendFlags(flags);
        }

        void setUserDataPtr(const T** user_var_)
        {
            Lock_t lock(this->mtx());
            m_user_var = user_var_;
        }

        IDataContainerConstPtr_t getData(const Header* desired = nullptr, IAsyncStream* stream = nullptr) override
        {
            auto data = TSubscriber<T>::getData(desired, stream);
            if (!data)
            {
                return data;
            }
            auto typed = std::static_pointer_cast<const TDataContainer<T>>(data);
            Lock_t lock(this->mtx());
            if (typed && m_user_var)
            {
                *m_user_var = typed->ptr();
            }
            return data;
        }

        std::ostream& print(std::ostream& os) const override
        {
            Lock_t lock(this->mtx());
            os << this->getTreeName();
            os << ' ';
            os << TypeInfo::create<T>();
            if (m_user_var)
            {
                if (*m_user_var)
                {
                    os << ' ';
                    os << **m_user_var;
                }
                else
                {
                    os << " unset";
                }
            }
            return os;
        }

        void onData(TDataContainerConstPtr_t<typename TSubscriber<T>::type> data,
                    const IParam& update_source,
                    UpdateFlags fg,
                    IAsyncStream& stream) override
        {
            TSubscriber<T>::onData(data, update_source, fg, stream);
            Lock_t lock(this->mtx());
            if (m_user_var)
            {
                *m_user_var = data->ptr();
            }
        }

      private:
        const T** m_user_var; // Pointer to the user space pointer variable of type T
    };

    // Deprecated in favor of ct::TArrayView
    template <typename T, class A>
    class MO_DEPRECATED TSubscriberPtr<std::vector<T, A>> : public TSubscriber<std::vector<T, A>>
    {
      public:
        using Input_t = const std::vector<T, A>*;

        TSubscriberPtr(const std::string& name = "", const std::vector<T, A>** user_var_ = nullptr)
            : m_user_var(user_var_)
        {
            this->setName(name);
        }

        void setUserDataPtr(const std::vector<T, A>** user_var_)
        {
            Lock_t lock(this->mtx());
            m_user_var = user_var_;
        }

        IDataContainerConstPtr_t getData(const Header* desired = nullptr, IAsyncStream* stream = nullptr) override
        {
            auto data = TSubscriber<std::vector<T, A>>::getData(desired, stream);
            if (data->getType().template isType<std::vector<T, A>>())
            {
                auto typed = std::static_pointer_cast<const TDataContainer<std::vector<T, A>>>(data);
                if (typed && m_user_var)
                {
                    *m_user_var = typed->ptr();
                }
            }
            return data;
        }

      protected:
        virtual void onData(TDataContainerConstPtr_t<std::vector<T, A>> data,
                            const IParam& param,
                            UpdateFlags fg,
                            IAsyncStream& stream)
        {
            Lock_t lock(this->mtx());
            TSubscriber<std::vector<T, A>>::onData(data, param, fg, stream);
            if (m_user_var)
            {
                *m_user_var = &data->data;
            }
        }

        const std::vector<T, A>** m_user_var; // Pointer to the user space pointer variable of type T
    };

    template <typename T>
    class MO_EXPORTS TSubscriberPtr<ct::TArrayView<T>> : virtual public TSubscriber<ct::TArrayView<T>>
    {
      public:
        using Input_t = ct::TArrayView<const T>;

        TSubscriberPtr(const std::string& name = "", ct::TArrayView<const T>* user_var_ = nullptr)
            : TSubscriber<ct::TArrayView<T>>(name)
            , TParam<ISubscriber>(name, ParamFlags::kINPUT)
            , IParam(name, ParamFlags::kINPUT)
            , m_user_var(user_var_)
        {
        }

        void setUserDataPtr(ct::TArrayView<const T>* user_var_)
        {
            Lock_t lock(this->mtx());
            m_user_var = user_var_;
        }

        IDataContainerPtr_t getData(const Header& desired = Header(),
                                    IAsyncStream& = IAsyncStream::currentRef()) override
        {
            auto data = TSubscriber<ct::TArrayView<T>>::getData(desired);
            if (data->getType() == TSubscriber<ct::TArrayView<T>>::getTypeInfo())
            {
                auto typed = std::static_pointer_cast<TDataContainer<ct::TArrayView<T>>>(data);
                if (typed && m_user_var)
                {
                    *m_user_var = typed->data;
                }
            }
            return data;
        }

        IDataContainerConstPtr_t getData(const Header& desired = Header(),
                                         IAsyncStream& = IAsyncStream::currentRef()) const override
        {
            return TSubscriber<ct::TArrayView<T>>::getData(desired);
        }

      protected:
        void updateDataImpl(const TDataContainerPtr_t<ct::TArrayView<T>>& data,
                            mo::UpdateFlags fg = UpdateFlags::kINPUT_UPDATED) override
        {
            Lock_t lock(this->mtx());
            TSubscriber<ct::TArrayView<T>>::updateDataImpl(data, fg);
            if (m_user_var)
            {
                *m_user_var = data->data;
            }
        }

        ct::TArrayView<const T>* m_user_var; // Pointer to the user space pointer variable of type T
    };

    template <typename T>
    class MO_EXPORTS TSubscriberPtr<std::shared_ptr<T>> : virtual public TSubscriber<T>
    {
      public:
        using Input_t = std::shared_ptr<const T>;

        TSubscriberPtr(const std::string& name = "", std::shared_ptr<const T>* user_var_ = nullptr)
            : TSubscriber<T>(name)
            , m_user_var(user_var_)
        {
        }

        void setUserDataPtr(std::shared_ptr<const T>* user_var_)
        {
            Lock_t lock(this->mtx());
            m_user_var = user_var_;
        }

        virtual void updateDataImpl(const TDataContainerPtr_t<T>& data)
        {
            Lock_t lock(this->mtx());
            TSubscriber<T>::updateDataImpl(data);
            if (m_user_var)
            {
                *m_user_var = data;
            }
        }

        IDataContainerPtr_t getData(const Header& desired = Header(),
                                    IAsyncStream& = IAsyncStream::currentRef()) override
        {
            auto data = TSubscriber<T>::getData(desired);
            auto typed = std::static_pointer_cast<TDataContainer<T>>(data);
            if (typed && m_user_var)
            {
                *m_user_var = typed->sharedPtr();
            }
            return data;
        }

        IDataContainerConstPtr_t getData(const Header& desired = Header(),
                                         IAsyncStream& = IAsyncStream::currentRef()) const override
        {
            return TSubscriber<T>::getData(desired);
        }

        std::ostream& print(std::ostream& os) const override
        {
            os << this->getTreeName();
            os << ' ';
            os << TypeInfo::create<T>();
            if (m_user_var)
            {
                if (*m_user_var)
                {
                    os << ' ';
                    os << **m_user_var;
                }
                else
                {
                    os << " unset";
                }
            }
            return os;
        }

      protected:
        std::shared_ptr<const T>* m_user_var; // Pointer to the user space pointer variable of type T
    };
} // namespace mo
