#pragma once

#include "StreamBuffer.hpp"

namespace mo
{
    namespace Buffer
    {
        template <class T>
        class MO_EXPORTS NNStreamBuffer : public StreamBuffer<T>
        {
          public:
            typedef typename ParamTraits<T>::Storage_t Storage_t;
            typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
            typedef typename ParamTraits<T>::InputStorage_t InputStorage_t;
            typedef typename ParamTraits<T>::Input_t Input_t;
            typedef void(TUpdateSig_t)(ConstStorageRef_t,
                                       IParam*,
                                       Context*,
                                       OptionalTime_t,
                                       size_t,
                                       const std::shared_ptr<ICoordinateSystem>&,
                                       UpdateFlags);
            typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
            typedef TSlot<TUpdateSig_t> TUpdateSlot_t;
            static const ParamType Type = NNStreamBuffer_e;
            typedef T ValueType;

            NNStreamBuffer(const std::string& name = "");

            virtual bool getData(InputStorage_t& data,
                                 const OptionalTime_t& ts = OptionalTime_t(),
                                 Context* ctx = nullptr,
                                 size_t* fn_ = nullptr);

            virtual bool
            getData(InputStorage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

            virtual ParamType getBufferType() const { return NNStreamBuffer_e; }

          protected:
            typename std::map<SequenceKey, InputStorage_t>::iterator search(const OptionalTime_t& ts);
            typename std::map<SequenceKey, InputStorage_t>::iterator search(size_t fn);
        };
    }

#define MO_METAParam_INSTANCE_NNBUFFER_(N)                                                                             \
    template <class T>                                                                                                 \
    struct MetaParam<T, N> : public MetaParam<T, N - 1, void>                                                          \
    {                                                                                                                  \
        static BufferConstructor<Buffer::NNStreamBuffer<T>> _nn_stream_buffer_constructor;                             \
        static ParamConstructor<Buffer::NNStreamBuffer<T>> _nn_stream_buffer_Param_constructor;                        \
        MetaParam<T, N>(const char* name) : MetaParam<T, N - 1>(name)                                                  \
        {                                                                                                              \
            (void)&_nn_stream_buffer_constructor;                                                                      \
            (void)&_nn_stream_buffer_Param_constructor;                                                                \
        }                                                                                                              \
    };                                                                                                                 \
    template <class T>                                                                                                 \
    BufferConstructor<Buffer::NNStreamBuffer<T>> MetaParam<T, N>::_nn_stream_buffer_constructor;                       \
    template <class T>                                                                                                 \
    ParamConstructor<Buffer::NNStreamBuffer<T>> MetaParam<T, N>::_nn_stream_buffer_Param_constructor;

    MO_METAParam_INSTANCE_NNBUFFER_(__COUNTER__)
}
#include "detail/NNStreamBufferImpl.hpp"
