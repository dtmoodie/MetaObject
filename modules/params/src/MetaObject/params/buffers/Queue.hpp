#pragma once
#include "BufferConstructor.hpp"
#include "IBuffer.hpp"
#include "MetaObject/detail/ConcurrentQueue.hpp"
#include "MetaObject/params/ITInputParam.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/ParamConstructor.hpp"

namespace mo
{
    class Context;
    namespace Buffer
    {
        template <class T>
        class Queue : public ITInputParam<T>, public IBuffer
        {
          public:
            typedef T ValueType;
            static const ParamType Type = Queue_e;

            Queue(const std::string& name);
            T* GetDataPtr(OptionalTime_t ts = OptionalTime_t(), Context* ctx = nullptr, size_t* fn_ = nullptr);
            T* GetDataPtr(size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

            T GetData(OptionalTime_t ts = OptionalTime_t(), Context* ctx = nullptr, size_t* fn = nullptr);
            T GetData(size_t fn, Context* ctx = nullptr, OptionalTime_t* ts = nullptr);

            bool GetData(T& value, OptionalTime_t ts = OptionalTime_t(), Context* ctx = nullptr, size_t* fn = nullptr);
            bool GetData(T& value, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts = nullptr);

            ITParam<T>* updateData(T& data_, mo::Time_t ts = -1 * mo::second, Context* ctx = nullptr);
            ITParam<T>* updateData(const T& data_, mo::Time_t ts = -1 * mo::second, Context* ctx = nullptr);
            ITParam<T>* updateData(T* data_, mo::Time_t ts = -1 * mo::second, Context* ctx = nullptr);

            bool Update(IParam* other, Context* ctx = nullptr);
            std::shared_ptr<IParam> DeepCopy() const;

            void SetSize(long long size);
            long long getSize();
            void getTimestampRange(mo::Time_t& start, mo::Time_t& end);
            virtual ParamType getBufferType() const { return Queue_e; }
          protected:
            virtual void onInputUpdate(Context* ctx, IParam* param);
            moodycamel::ConcurrentQueue<T> _queue;
        };
    }
}
