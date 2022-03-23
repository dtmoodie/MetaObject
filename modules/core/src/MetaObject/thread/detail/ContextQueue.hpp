#ifndef MO_THREAD_CONTEXT_QUEUE_HPP
#define MO_THREAD_CONTEXT_QUEUE_HPP

#include <MetaObject/thread/fiber_include.hpp>

#include "../PriorityLevels.hpp"
#include "context_spinlock_queue.hpp"

namespace mo
{
    class ContextWorkQueue
    {
      public:
        uint64_t push(boost::fibers::context* c, const PriorityLevels priority);
        boost::fibers::context* pop();
        boost::fibers::context* steal();

        bool empty() const;

      private:
        mo::fibers::detail::SpinlockQueue<boost::fibers::context> m_queues[HIGHEST + 1];
    };
} // namespace mo

#endif // MO_THREAD_CONTEXT_QUEUE_HPP
