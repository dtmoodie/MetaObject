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
    struct MO_EXPORTS TInputParamPtr : virtual public ITInputParam<T>
    {
        using ContainerPtr_t = typename ITParam<T>::ContainerPtr_t;
        using TUpdateSlot_t = typename ITParam<T>::TUpdateSlot_t;

        TInputParamPtr(const std::string& name = "", T** user_var_ = nullptr);

        bool setInput(std::shared_ptr<IParam> input);
        bool setInput(IParam* input);
        void setUserDataPtr(T** user_var_);
        bool getInput(const OptionalTime_t& ts, size_t* fn = nullptr);
        bool getInput(size_t fn, OptionalTime_t* ts = nullptr);

        virtual ConstAccessToken<T> read() const;
        bool canAccess() const override
        {
            return ParamTraits<T>::valid(_current_data);
        }

      protected:
        void onInputUpdate(ContainerPtr_t data, IParam* param, UpdateFlags fg) override
        {
            Lock lock(this->mtx());
            const auto header = data->getHeader();
            if (fg == mo::BufferUpdated_e && param->checkFlags(mo::ParamFlags::Buffer_e))
            {
                emitTypedUpdate(data, this, header, InputUpdated_e);
                emitUpdate(this, header, fg, InputUpdated_e);
                return;
            }
            if (header.ctx == this->_header.ctx)
            {
                _current_data = data;
                if (_user_var)
                {
                    *_user_var = &data->data;
                    emitTypedUpdate(data, this, header, InputUpdated_e);
                    emitUpdate(this, header, InputUpdated_e);
                }
            }
        }

      private:
        T** _user_var; // Pointer to the user space pointer variable of type T
        ContainerPtr_t _current_data;
    };
}
