#include "AsyncStream.hpp"
#include "common.hpp"

#include "MetaObject/logging/logging.hpp"
#include <MetaObject/logging/profiling.hpp>
#include <cuda_runtime_api.h>

namespace mo
{
    namespace cuda
    {

        AsyncStream::AsyncStream()
            : mo::AsyncStream(TypeInfo(typeid(cuda::IAsyncStream)))
        {
            init();
        }

        AsyncStream::AsyncStream(TypeInfo type)
            : mo::AsyncStream(type)
        {
            init();
        }

        AsyncStream::AsyncStream(const Stream& stream)
            : m_stream(stream)
        {
            init();
        }

        AsyncStream::~AsyncStream()
        {
            m_stream.synchronize();
        }

        void AsyncStream::init()
        {
            CUDA_ERROR_CHECK(cudaGetDevice(&m_device_id));
        }

        void AsyncStream::setName(const std::string& name)
        {
            setStreamName(name.c_str(), getStream());
            mo::AsyncStream::setName(name);
        }

        void AsyncStream::setDevicePriority(const PriorityLevels lvl)
        {
        }

        Stream AsyncStream::getStream() const
        {
            return m_stream;
        }
    }
}
