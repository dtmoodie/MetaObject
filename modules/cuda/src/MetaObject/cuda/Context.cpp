#include "Context.hpp"
#if MO_HAVE_CUDA
#include "MetaObject/logging/logging.hpp"
#include <MetaObject/logging/profiling.hpp>
#include <cuda_runtime_api.h>
#ifdef MO_HAVE_OPENCV
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

namespace mo
{
    namespace cuda
    {

        Context::Context()
            : mo::Context(TypeInfo(typeid(cuda::Context)))
            , m_priority(5)
        {
            init();
        }

        Context::Context(TypeInfo type, const int32_t priority)
            : mo::Context(type)
            , m_priority(priority)
        {
            init();
        }

        Context::Context(cudaStream_t stream)
            : m_cuda_stream(stream)
            , m_owns_stream(false)
        {
        }

        Context::~Context()
        {
            {
                CUDA_ERROR_CHECK(cudaStreamSynchronize(this->m_cuda_stream));
            }
            {
                if (m_owns_stream)
                {
                    CUDA_ERROR_CHECK(cudaStreamDestroy(this->m_cuda_stream));
                }
            }
            this->m_cuda_stream = nullptr;
        }

        void Context::init()
        {
            {
                CUDA_ERROR_CHECK(cudaStreamCreateWithPriority(&m_cuda_stream, cudaStreamNonBlocking, m_priority))
                    << " error creating stream";
            }
            {
                int device_id;
                CUDA_ERROR_CHECK(cudaGetDevice(&device_id));
                setDeviceId(device_id);
            }
        }

        void Context::setName(const std::string& name)
        {
            setStreamName(name.c_str(), stream());
            mo::Context::setName(name);
        }

        cudaStream_t Context::stream() const
        {
            return m_cuda_stream;
        }

        void Context::setStream(cudaStream_t stream)
        {
            if (m_cuda_stream && m_owns_stream)
            {
                CUDA_ERROR_CHECK(cudaStreamDestroy(this->m_cuda_stream)) << "Unable to cleanup stream";
            }
            m_cuda_stream = stream;
            m_owns_stream = false;
        }
    }
}
#endif
