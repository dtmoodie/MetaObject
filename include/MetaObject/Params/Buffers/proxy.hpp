#pragma once
#include "MetaObject/Params/ITParam.hpp"
namespace mo
{
    template<typename T> class ITParam;
    namespace Buffer
    {
        class IBuffer;
        template<typename T> class Proxy : public ITParam<T>, public IBuffer
        {
            ITParam<T>* _input_Param;
            std::shared_ptr<ITParam<T>> _buffer;

            std::shared_ptr<Signals::Connection> _input_update_Connection;
            std::shared_ptr<Signals::Connection> _input_delete_Connection;
        public:
            Proxy(ITParam<T>* input, Params::Param::Ptr buffer) :
                _input_Param(input),
                _buffer(std::dynamic_pointer_cast<ITParam>(buffer)),
                ITParam<T>("proxy for " + (input ? input->getName() : std::string("null")))
            {
                if (input)
                {
                    _input_update_Connection = _input_Param->RegisterNotifier(std::bind(&Proxy::onInputUpdate, this, std::placeholders::_1));
                    _input_delete_Connection = _input_Param->registerDeleteNotifier(std::bind(&Proxy::onInputDelete, this));
                    _input_Param->subscribers++;
                }
            }
            Proxy(ITParam<T>* input, ITParam<T>* buffer) :
                _input_Param(input),
                _buffer(buffer),
                ITParam<T>("proxy")

            {
                if (input)
                {
                    _input_update_Connection = _input_Param->RegisterNotifier(std::bind(&Proxy::onInputUpdate, this, std::placeholders::_1));
                    _input_delete_Connection = _input_Param->registerDeleteNotifier(std::bind(&Proxy::onInputDelete, this));
                    _input_Param->subscribers++;
                }
            }
            ~Proxy()
            {
                if (_input_Param)
                    _input_Param->subscribers--;
                _input_Param = nullptr;
                _input_update_Connection.reset();
                _input_delete_Connection.reset();
            }
            void setInput(Param* param)
            {
                if (_input_Param = dynamic_cast<ITParam<T>*>(param))
                {
                    _input_update_Connection = _input_Param->RegisterNotifier(std::bind(&Proxy::onInputUpdate, this, std::placeholders::_1));
                    _input_delete_Connection = _input_Param->registerDeleteNotifier(std::bind(&Proxy::onInputDelete, this));
                }
            }
            void onInputDelete()
            {
                _input_update_Connection.reset();
                _input_delete_Connection.reset();
                _input_Param = nullptr;
            }
            void onInputUpdate(cv::cuda::Stream* stream)
            {
                if (_input_Param)
                {
                    auto time = _input_Param->GetTimeIndex();
                    if (auto data = _input_Param->Data(time))
                    {
                        _buffer->UpdateData(data, time, stream);
                    }
                }
            }
            virtual T* Data(mo::Time_t timestamp)
            {
                return _buffer->Data(timestamp);
            }
            virtual bool GetData(T& value, mo::Time_t ts = -1 * mo::second)
            {
                return _buffer->GetData(value, time_index);
            }
            virtual void UpdateData(T& data_, mo::Time_t ts = -1 * mo::second, cv::cuda::Stream* stream = nullptr)
            {
            }
            virtual void UpdateData(const T& data_, mo::Time_t ts = -1 * mo::second, cv::cuda::Stream* stream = nullptr)
            {    
            }
            virtual void UpdateData(T* data_, mo::Time_t ts = -1 * mo::second, cv::cuda::Stream* stream = nullptr)
            {
            }

            virtual Param::Ptr DeepCopy() const
            {
                return Param::Ptr(new Proxy(_input_Param, _buffer->DeepCopy()));
            }
            virtual void SetSize(mo::Time_t size)
            {
                dynamic_cast<IBuffer*>(_buffer.get())->SetSize(size);
            }
            virtual mo::Time_t GetSize()
            {
                return dynamic_cast<IBuffer*>(_buffer.get())->GetSize();
            }
            virtual void getTimestampRange(mo::Time_t& start, mo::Time_t& end)
            {
                dynamic_cast<IBuffer*>(_buffer.get())->getTimestampRange(start, end);
            }
            virtual std::recursive_mutex& mtx()
            {
                return _buffer->mtx();
            }
        };
    }
}