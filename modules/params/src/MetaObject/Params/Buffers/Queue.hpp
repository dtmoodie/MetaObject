#pragma once
#include "MetaObject/Detail/ConcurrentQueue.hpp"
#include "MetaObject/Params/ITInputParam.hpp"
#include "MetaObject/Params/ParamConstructor.hpp"
#include "MetaObject/Params/MetaParam.hpp"
#include "BufferConstructor.hpp"
#include "IBuffer.hpp"

namespace mo
{
    class Context;
    namespace Buffer
    {
        template<class T>
        class Queue: public ITInputParam<T>, public IBuffer
        {
        public:
            typedef T ValueType;
            static const ParamType Type = Queue_e;

            Queue(const std::string& name);
            T*   GetDataPtr(OptionalTime_t ts = OptionalTime_t(),
                                    Context* ctx = nullptr, size_t* fn_ = nullptr);
            T*   GetDataPtr(size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

            T    GetData(OptionalTime_t ts = OptionalTime_t(),
                                 Context* ctx = nullptr, size_t* fn = nullptr);
            T    GetData(size_t fn, Context* ctx = nullptr, OptionalTime_t* ts = nullptr);

            bool GetData(T& value, OptionalTime_t ts = OptionalTime_t(),
                                 Context* ctx = nullptr, size_t* fn = nullptr);
            bool GetData(T& value, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts = nullptr);

            ITParam<T>* UpdateData(T& data_, mo::Time_t ts = -1 * mo::second, Context* ctx = nullptr);
            ITParam<T>* UpdateData(const T& data_, mo::Time_t ts = -1 * mo::second, Context* ctx = nullptr);
            ITParam<T>* UpdateData(T* data_, mo::Time_t ts = -1 * mo::second, Context* ctx = nullptr);

            bool Update(IParam* other, Context* ctx = nullptr);
            std::shared_ptr<IParam> DeepCopy() const;

            void SetSize(long long size);
            long long GetSize();
            void getTimestampRange(mo::Time_t& start, mo::Time_t& end);
            virtual ParamType GetBufferType() const{ return Queue_e;}
        protected:
            virtual void onInputUpdate(Context* ctx, IParam* param);
            moodycamel::ConcurrentQueue<T> _queue;
        };
    }
}
