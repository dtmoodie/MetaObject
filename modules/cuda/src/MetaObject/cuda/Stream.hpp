#ifndef MO_CUDA_STREAM_HPP
#define MO_CUDA_STREAM_HPP
#include <MetaObject/detail/Export.hpp>

#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace mo
{
    namespace cuda
    {
        struct Event;
        struct MO_EXPORTS Stream
        {
            using Ptr_t = std::shared_ptr<CUstream_st>;

            static Ptr_t create(int priority = 0);
            static Stream& null();

            Stream(cudaStream_t stream, bool owns_stream = false);
            Stream(int priority = 0);
            Stream(Ptr_t stream);

            operator cudaStream_t();
            operator CUstream_st const*() const;

            void waitEvent(Event& event);
            void synchronize() const;
            bool query() const;

          private:
            Ptr_t m_stream;
        };
    } // namespace cuda
} // namespace mo

#endif // MO_CUDA_STREAM_HPP
