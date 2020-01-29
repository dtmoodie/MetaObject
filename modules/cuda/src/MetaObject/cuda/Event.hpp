#ifndef MO_CUDA_EVENT_HPP
#define MO_CUDA_EVENT_HPP
#include <MetaObject/core/detail/ObjectPool.hpp>
#include <MetaObject/core/detail/Time.hpp>

#include <memory>

struct CUevent_st;
using cudaEvent_t = CUevent_st*;
using constCudaEvent_t = CUevent_st const*;

namespace mo
{
    namespace cuda
    {
        struct Stream;

        struct MO_EXPORTS Event
        {
            static std::shared_ptr<CUevent_st> create();

            Event(ObjectPool<CUevent_st>* event_pool);

            void record(Stream& stream);

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
            bool synchronize(Duration timeout = 0 * ms) const;

            /**
             * @brief setCallback to be called on event completion
             */
            void setCallback(std::function<void(void)>&& cb);

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
