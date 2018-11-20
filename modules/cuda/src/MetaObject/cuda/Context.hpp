#pragma once
#include <MetaObject/core/AsyncStream.hpp>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace mo
{
    namespace cuda
    {
        struct IContext : virtual public mo::IContext
        {
            virtual cudaStream_t stream() const = 0;
            virtual void setStream(cudaStream_t stream) = 0;

        }; // struct mo::cuda::Context

        struct Context : virtual public cuda::IContext, public mo::Context
        {
            Context();
            ~Context();

            cudaStream_t stream() const override;

            void setName(const std::string& name) override;
            void setStream(cudaStream_t stream) override;

          protected:
            Context(TypeInfo type, const int32_t priority);
            Context(cudaStream_t stream);

          private:
            void init();

            int32_t m_priority;
            cudaStream_t m_cuda_stream;
            bool m_owns_stream;
        }; // struct mo::cuda::Context

    } // namespace mo::cuda
} // namespace mo
