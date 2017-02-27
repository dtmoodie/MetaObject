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
            T*   GetDataPtr(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                                    Context* ctx = nullptr, size_t* fn_ = nullptr);
            T*   GetDataPtr(size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts_ = nullptr);

            T    GetData(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                                 Context* ctx = nullptr, size_t* fn = nullptr);
            T    GetData(size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts = nullptr);

            bool GetData(T& value, boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                                 Context* ctx = nullptr, size_t* fn = nullptr);
            bool GetData(T& value, size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts = nullptr);

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
