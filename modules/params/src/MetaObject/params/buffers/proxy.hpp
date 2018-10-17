#pragma once
#include "MetaObject/params/ITParam.hpp"
namespace mo
{
    namespace Buffer
    {
        class IBuffer;
        template <typename T>
        class Proxy : public TParam<T>, public IBuffer
        {
            TParam<T>* _input_Param;
            std::shared_ptr<TParam<T>> _buffer;

            std::shared_ptr<Signals::Connection> _input_update_Connection;
            std::shared_ptr<Signals::Connection> _input_delete_Connection;

          public:
            Proxy(TParam<T>* input, Params::Param::Ptr buffer)
                : _input_Param(input), _buffer(std::dynamic_pointer_cast<TParam>(buffer)),
                  TParam<T>("proxy for " + (input ? input->getName() : std::string("null")))
            {
                if (input)
                {
                    _input_update_Connection =
                        _input_Param->RegisterNotifier(std::bind(&Proxy::onInputUpdate, this, std::placeholders::_1));
                    _input_delete_Connection =
                        _input_Param->registerDeleteNotifier(std::bind(&Proxy::onInputDelete, this));
                    _input_Param->subscribers++;
                }
            }
            Proxy(TParam<T>* input, TParam<T>* buffer) : _input_Param(input), _buffer(buffer), TParam<T>("proxy")

            {
                if (input)
                {
                    _input_update_Connection =
                        _input_Param->RegisterNotifier(std::bind(&Proxy::onInputUpdate, this, std::placeholders::_1));
                    _input_delete_Connection =
                        _input_Param->registerDeleteNotifier(std::bind(&Proxy::onInputDelete, this));
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
                if (_input_Param = dynamic_cast<TParam<T>*>(param))
                {
                    _input_update_Connection =
                        _input_Param->RegisterNotifier(std::bind(&Proxy::onInputUpdate, this, std::placeholders::_1));
                    _input_delete_Connection =
                        _input_Param->registerDeleteNotifier(std::bind(&Proxy::onInputDelete, this));
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
                        _buffer->updateData(data, time, stream);
                    }
                }
            }
            virtual T* Data(mo::Time timestamp) { return _buffer->Data(timestamp); }
            virtual bool GetData(T& value, mo::Time ts = -1 * mo::second)
            {
                return _buffer->GetData(value, time_index);
            }
            virtual void updateData(T& data_, mo::Time ts = -1 * mo::second, cv::cuda::Stream* stream = nullptr) {}
            virtual void updateData(const T& data_, mo::Time ts = -1 * mo::second, cv::cuda::Stream* stream = nullptr)
            {
            }
            virtual void updateData(T* data_, mo::Time ts = -1 * mo::second, cv::cuda::Stream* stream = nullptr) {}

            virtual Param::Ptr DeepCopy() const { return Param::Ptr(new Proxy(_input_Param, _buffer->DeepCopy())); }
            virtual void SetSize(mo::Time size) { dynamic_cast<IBuffer*>(_buffer.get())->SetSize(size); }
            virtual mo::Time getSize() { return dynamic_cast<IBuffer*>(_buffer.get())->getSize(); }
            virtual void getTimestampRange(mo::Time& start, mo::Time& end)
            {
                dynamic_cast<IBuffer*>(_buffer.get())->getTimestampRange(start, end);
            }
            virtual std::recursive_mutex& mtx() { return _buffer->mtx(); }
        };
    }
}
