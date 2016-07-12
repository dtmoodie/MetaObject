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

#include "MetaObject/Parameters/ITypedParameter.hpp"
#include "MetaObject/Parameters/ParameterConstructor.hpp"
#include "IBuffer.hpp"
#include <boost/circular_buffer.hpp>

namespace mo
{
    namespace Buffer
    {
        template<typename T> class CircularBuffer: public ITypedParameter<T>, public IBuffer
        {
            static ParameterConstructor<CircularBuffer<T>, T, CircularBuffer_e> _circular_buffer_constructor;
            boost::circular_buffer<std::pair<long long, T>> _data_buffer;
        public:
            CircularBuffer(const std::string& name = "",
                const T& init = T(), long long ts = -1,
                ParameterType type = Buffer):
                ITypedParameter<T>(name)
            {
                (void)&_circular_buffer_constructor;
                _data_buffer.set_capacity(10);
                _data_buffer.push_back(std::make_pair(time_index, init));
            }
            virtual T* Data(long long time_index = -1)
            {
                if (time_index == -1 && _data_buffer.size())
                    return &_data_buffer.back().second;

                for (auto& itr : _data_buffer)
                {
                    if (itr.first == time_index)
                    {
                        return &itr.second;
                    }
                }
                return nullptr;
            }
            virtual bool GetData(T& value, long long time_index = -1)
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                if (time_index == -1 && _data_buffer.size())
                {
                    value = _data_buffer.back().second;
                    return true;
                }
                for (auto& itr : _data_buffer)
                {
                    if (itr.first == time_index)
                    {
                        value = itr.second;
                        return true;
                    }
                }
                return false;
            }
            virtual void UpdateData(T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                _data_buffer.push_back(std::pair<long long, T>(time_index, data_));
                Parameter::changed = true;
                Parameter::OnUpdate(stream);
            }
            virtual void UpdateData(const T& data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                _data_buffer.push_back(std::pair<long long, T>(time_index, data_));
                Parameter::changed = true;
                Parameter::OnUpdate(stream);
            }
            virtual void UpdateData(T* data_, long long time_index = -1, cv::cuda::Stream* stream = nullptr)
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                _data_buffer.push_back(std::pair<long long, T>(time_index, *data_));
                Parameter::changed = true;
                Parameter::OnUpdate(stream);
            }

            virtual Loki::TypeInfo GetTypeInfo()
            {
                return Loki::TypeInfo(typeid(T));
            }
            virtual bool Update(Parameter::Ptr other, cv::cuda::Stream* stream = nullptr)
            {
                auto typedParameter = std::dynamic_pointer_cast<ITypedParameter<T>>(other);
                if (typedParameter)
                {
                    auto ptr = typedParameter->Data();
                    if (ptr)
                    {
                        std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                        _data_buffer.push_back(std::pair<long long, T>(typedParameter->GetTimeIndex(), *ptr));
                        Parameter::changed = true;
                        Parameter::OnUpdate(stream);
                    }
                }
                return false;
            }
            virtual void SetSize(long long size)
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                _data_buffer.set_capacity(size);
            }
            virtual long long GetSize() 
            {
                std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                return _data_buffer.capacity();
            }
            virtual void GetTimestampRange(long long& start, long long& end) 
            {
                if (_data_buffer.size())
                {
                    std::lock_guard<std::recursive_mutex> lock(Parameter::_mtx);
                    start = _data_buffer.back().first;
                    end = _data_buffer.front().first;
                }
            }
            virtual Parameter::Ptr DeepCopy() const
            {
                return Parameter::Ptr(new CircularBuffer<T>(Parameter::name));
            }
        };
        template<typename T> FactoryRegisterer<CircularBuffer<T>, T, CircularBuffer_c> CircularBuffer<T>::_circular_buffer_constructor;
    }
}
