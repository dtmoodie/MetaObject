#include "Stream.hpp"
#include "Event.hpp"
#include "errors.hpp"

#include <cuda_runtime_api.h>

namespace mo
{
    namespace cuda
    {
        Stream& Stream::null()
        {
            thread_local Stream stream(std::shared_ptr<CUstream_st>(nullptr, [](cudaStream_t) {}));
            return stream;
        }

        std::shared_ptr<CUstream_st> Stream::create(const int priority)
        {
            cudaStream_t stream = nullptr;
            CHECK_CUDA_ERROR(&cudaStreamCreateWithPriority, &stream, cudaStreamNonBlocking, priority);
            return std::shared_ptr<CUstream_st>(stream, [](cudaStream_t st) {
                CHECK_CUDA_ERROR(&cudaStreamSynchronize, st);
                CHECK_CUDA_ERROR(&cudaStreamDestroy, st);
            });
        }

        Stream::Stream(cudaStream_t stream, bool owns_stream)
        {
            if (owns_stream)
            {
                m_stream = Ptr_t(stream, [](cudaStream_t st) {
                    CHECK_CUDA_ERROR(&cudaStreamSynchronize, st);
                    CHECK_CUDA_ERROR(&cudaStreamDestroy, st);
                });
            }
            else
            {
                m_stream = Ptr_t(stream, [](cudaStream_t) {});
            }
        }

        Stream::Stream(const int priority)
            : m_stream(create(priority))
        {
        }

        Stream::Stream(Ptr_t stream)
            : m_stream(std::move(stream))
        {
        }

        Stream::operator cudaStream_t()
        {
            return m_stream.get();
        }

        Stream::operator CUstream_st const*() const
        {
            return m_stream.get();
        }

        void Stream::waitEvent(Event& event)
        {
            if (!event.queryCompletion())
            {
                CHECK_CUDA_ERROR(&cudaStreamWaitEvent, m_stream.get(), event, 0);
            }
        }

        void Stream::synchronize() const
        {
            CHECK_CUDA_ERROR(&cudaStreamSynchronize, m_stream.get());
        }

        bool Stream::query() const
        {
            return cudaStreamQuery(m_stream.get()) == cudaSuccess;
        }
    }
}
