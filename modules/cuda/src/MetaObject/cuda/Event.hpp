#ifndef MO_CUDA_EVENT_HPP
#define MO_CUDA_EVENT_HPP
#include <MetaObject/core/detail/ObjectPool.hpp>
#include <MetaObject/core/detail/Time.hpp>

#include <memory>

struct CUevent_st;
struct CUstream_st;
using cudaEvent_t = CUevent_st*;
using constCudaEvent_t = CUevent_st const*;
using cudaStream_t = CUstream_st*;

namespace mo
{
    namespace cuda
    {
        struct Stream;

        struct MO_EXPORTS Event
        {
            static std::shared_ptr<CUevent_st> create();

            Event(std::shared_ptr<ObjectPool<CUevent_st>> event_pool);
            ~Event();

            void record(Stream& stream);
            void record(cudaStream_t stream);

            /**
             * @brief queryCompletion check if the event has triggered
             * @return true if complete, false if not complete
             */
            bool queryCompletion() const;

            /**
             * @brief synchronize blocks the current CPU fiber until even it complete.
             *        effectively this is a while(!queryCompletion()){boost::fiber::yield();}
             * @param timeout how long to wait, 0 = infinity
             * @return true on completion, false on timeout exceeded
             */
            bool synchronize(Duration sleep = 1 * ns, Duration timeout = 0 * ms) const;

            /**
             * @brief setCallback to be called on event completion. Callback can be called from ANY thread that services
             * this event
             * @brief priority is the priority of executing the callback
             */
            void setCallback(std::function<void(mo::IAsyncStream*)>&& cb, uint64_t event_id = 0);

            operator cudaEvent_t();
            operator constCudaEvent_t() const;

          private:
            // Private implementation to prevent mass inclusion of boost fiber
            // need to hide boost fiber from nvcc otherwise it will fail to compile on many platforms
            struct Impl;
            std::shared_ptr<Impl> m_impl;
        };
    } // namespace cuda

    template <>
    struct ObjectConstructor<CUevent_st>
    {
        using Ptr_t = std::shared_ptr<CUevent_st>;

        Ptr_t createShared()
        {
            return cuda::Event::create();
        }
    };
} // namespace mo

#endif // MO_CUDA_EVENT_HPP
