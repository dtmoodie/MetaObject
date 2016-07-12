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

https://github.com/dtmoodie/parameters
*/
#pragma once

#include "parameters/ITypedParameter.hpp"
#include "IBuffer.hpp"
#include <map>

namespace Parameters
{
    namespace Buffer
    {
        template<typename T> class Map: public ITypedParameter<T>, public IBuffer
        {
        protected:
            std::map<long long, T> _data_buffer;
        public:
            Map(const std::string& name,
                const T& init = T(), long long time_index = -1,
                ParameterType type = Buffer) :
                ITypedParameter<T>(name, type)
            {
            }
            virtual T* GetData(long long time_index = -1, Signals::context* ctx = nullptr)
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                if (time_index == -1 && _data_buffer.size())
                {
                    return &(_data_buffer.rbegin()->second);
                }
                else
                {
                    auto itr = _data_buffer.find(time_index);
                    if (itr != _data_buffer.end())
                    {
                        return &itr->second;
                    }
                }
                return nullptr;
            }
            virtual bool GetData(T& value, long long time_index = -1, Signals::context* ctx = nullptr)
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                if (time_index == -1 && _data_buffer.size())
                {
                    value = _data_buffer.rbegin()->second;
                    return true;
                }
                auto itr = _data_buffer.find(time_index);
                if (itr != _data_buffer.end())
                {
                    value = itr->second;
                }
                return false;
            }
            virtual ITypedParameter<T>* UpdateData(T& data_, long long time_index = -1, Signals::context* ctx = nullptr)
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                _data_buffer[time_index] = data_;
                Parameter::changed = true;
                Parameter::OnUpdate(stream);
            }
            virtual ITypedParameter<T>* UpdateData(const T& data_, long long time_index = -1, Signals::context* ctx = nullptr)
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                _data_buffer[time_index] = data_;
                Parameter::changed = true;
                Parameter::OnUpdate(stream);
            }
            virtual ITypedParameter<T>* UpdateData(T* data_, long long time_index = -1, Signals::context* ctx = nullptr)
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                _data_buffer[time_index] = *data_;
                Parameter::changed = true;
                Parameter::OnUpdate(stream);
            }

            virtual Loki::TypeInfo GetTypeInfo() const
            {
                return Loki::TypeInfo(typeid(T));
            }
            virtual bool Update(Parameter* other, Signals::context* ctx = nullptr)
            {
                auto typedParameter = std::dynamic_pointer_cast<ITypedParameter<T>*>(other);
                if (typedParameter)
                {
                    auto ptr = typedParameter->Data();
                    if (ptr)
                    {
                        std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                        _data_buffer[typedParameter->GetTimeIndex()] = *ptr;
                        Parameter::changed = true;
                        Parameter::OnUpdate(stream);
                    }
                }
                return false;
            }
            virtual void SetSize(long long size)
            {
                
            }
            virtual long long GetSize() 
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                return _data_buffer.size();
            }
            virtual void GetTimestampRange(long long& start, long long& end) 
            {
                if (_data_buffer.size())
                {
                    std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                    start = _data_buffer.begin()->first;
                    end = _data_buffer.rbegin()->first;
                }
            }
            virtual Parameter* DeepCopy() const
            {
                auto ptr = new Map<T>(Parameter::name);
                ptr->_data_buffer = this->_data_buffer;
                return Parameter::Ptr(ptr);
            }
        };
    }
}
