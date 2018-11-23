#ifndef MO_THREAD_CONTEXT_QUEUE_HPP
#define MO_THREAD_CONTEXT_QUEUE_HPP

#include "../PriorityLevels.hpp"
#include "context_spinlock_queue.hpp"

namespace mo
{
    class ContextQueue
    {
      public:
        uint64_t push(boost::fibers::context* c, const PriorityLevels priority);
        boost::fibers::context* pop();
        boost::fibers::context* steal();

        bool empty() const;

      private:
        mo::fibers::detail::context_spinlock_queue m_queues[HIGHEST + 1];
    };
}

#endif // MO_THREAD_CONTEXT_QUEUE_HPP
