#pragma once
#include "BufferConstructor.hpp"
#include "IBuffer.hpp"
#include "MetaObject/detail/ConcurrentQueue.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/ParamConstructor.hpp"
#include "MetaObject/params/TSubscriber.hpp"

namespace mo
{
    class Context;
    namespace Buffer
    {
        template <class T>
        class Queue : public TSubscriber<T>, public IBuffer
        {
          public:
            typedef T ValueType;
            static const BufferFlags Type = Queue_e;

            Queue(const std::string& name);
            T* GetDataPtr(OptionalTime ts = OptionalTime(), Context* ctx = nullptr, size_t* fn_ = nullptr);
            T* GetDataPtr(size_t fn, Context* ctx = nullptr, OptionalTime* ts_ = nullptr);

            T GetData(OptionalTime ts = OptionalTime(), Context* ctx = nullptr, size_t* fn = nullptr);
            T GetData(size_t fn, Context* ctx = nullptr, OptionalTime* ts = nullptr);

            bool GetData(T& value, OptionalTime ts = OptionalTime(), Context* ctx = nullptr, size_t* fn = nullptr);
            bool GetData(T& value, size_t fn, Context* ctx = nullptr, OptionalTime* ts = nullptr);

            TParam<T>* updateData(T& data_, mo::Time ts = -1 * mo::second, Context* ctx = nullptr);
            TParam<T>* updateData(const T& data_, mo::Time ts = -1 * mo::second, Context* ctx = nullptr);
            TParam<T>* updateData(T* data_, mo::Time ts = -1 * mo::second, Context* ctx = nullptr);

            bool Update(IParam* other, Context* ctx = nullptr);
            std::shared_ptr<IParam> DeepCopy() const;

            void SetSize(long long size);
            long long getSize();
            void getTimestampRange(mo::Time& start, mo::Time& end);
            virtual BufferFlags getBufferType() const
            {
                return Queue_e;
            }

          protected:
            virtual void onInputUpdate(Context* ctx, IParam* param);
            moodycamel::ConcurrentQueue<T> _queue;
        };
    } // namespace Buffer
} // namespace mo
