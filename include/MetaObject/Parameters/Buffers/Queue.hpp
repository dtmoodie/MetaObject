#pragma once
#include "MetaObject/Detail/ConcurrentQueue.hpp"
#include "MetaObject/Parameters/ITypedInputParameter.hpp"
#include "MetaObject/Parameters/ParameterConstructor.hpp"
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "BufferConstructor.hpp"
#include "IBuffer.hpp"

namespace mo
{
    class Context;
    namespace Buffer
    {
        template<class T>
        class Queue: public ITypedInputParameter<T>, public IBuffer
        {
        public:
            typedef T ValueType;
            static const ParameterTypeFlags Type = Queue_e;

            Queue(const std::string& name);
            T*   GetDataPtr(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr);
            bool GetData(T& value, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr);
            T    GetData(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr);

            ITypedParameter<T>* UpdateData(T& data_, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr);
            ITypedParameter<T>* UpdateData(const T& data_, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr);
            ITypedParameter<T>* UpdateData(T* data_, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr);

            bool Update(IParameter* other, Context* ctx = nullptr);
            std::shared_ptr<IParameter> DeepCopy() const;

            void SetSize(long long size);
            long long GetSize();
            void GetTimestampRange(mo::time_t& start, mo::time_t& end);
            virtual ParameterTypeFlags GetBufferType() const{ return Queue_e;}
        protected:
            virtual void onInputUpdate(Context* ctx, IParameter* param);
            moodycamel::ConcurrentQueue<T> _queue;
        };
    }
}
