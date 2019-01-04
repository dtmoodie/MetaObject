#ifndef MO_CUDA_STREAM_HPP
#define MO_CUDA_STREAM_HPP
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace mo
{
    namespace cuda
    {
        struct Event;
        struct Stream
        {
            using Ptr_t = std::shared_ptr<CUstream_st>;

            static Ptr_t create(const int priority = 0);
            static Stream& null();

            Stream(cudaStream_t stream, bool owns_stream = false);
            Stream(const int priority = 0);
            Stream(const Ptr_t& stream);

            operator cudaStream_t();
            operator cudaStream_t const() const;

            void waitEvent(Event& event);
            void synchronize() const;
            bool query() const;

          private:
            Ptr_t m_stream;
        };
    }
}

#endif // MO_CUDA_STREAM_HPP
