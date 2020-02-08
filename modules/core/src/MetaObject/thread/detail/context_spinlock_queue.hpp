
//          Copyright Oliver Kowalke 2015.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef MO_FIBERS_DETAIL_SPINLOCK_QUEUE_H
#define MO_FIBERS_DETAIL_SPINLOCK_QUEUE_H

#include <cstddef>
#include <cstring>
#include <mutex>

#include <boost/config.hpp>

#include <boost/fiber/context.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

namespace mo
{
    namespace fibers
    {
        namespace detail
        {
            struct DefaultContextStealPolicy
            {
                static bool canSteal(boost::fibers::context* ctx)
                {
                    return !ctx->is_context(boost::fibers::type::pinned_context);
                }
            };

            template <class T = boost::fibers::context, class STEAL_POLICY = DefaultContextStealPolicy>
            struct RobbableQueue
            {
                RobbableQueue(const std::size_t capacity = 4096)
                    : m_push_idx(0)
                    , m_current_idx(0)
                    , m_capacity(capacity)
                {
                    m_slots = new T*[m_capacity];
                }

                ~RobbableQueue()
                {
                    delete[] m_slots;
                }

                RobbableQueue(RobbableQueue const&) = delete;
                RobbableQueue& operator=(RobbableQueue const&) = delete;

                std::size_t size() const noexcept
                {
                    if (m_current_idx > m_push_idx)
                    {
                        return (m_capacity - m_current_idx) + m_push_idx;
                    }
                    else
                    {
                        return (m_push_idx - m_current_idx);
                    }
                }

                bool isFull() const noexcept
                {
                    return m_current_idx == ((m_push_idx + 1) % m_capacity);
                }

                void push(T* c)
                {
                    if (isFull())
                    {
                        resize();
                    }
                    m_slots[m_push_idx] = c;
                    m_push_idx = (m_push_idx + 1) % m_capacity;
                }

                bool isEmpty() const noexcept
                {
                    return m_current_idx == m_push_idx;
                }

                T* pop()
                {
                    T* c = nullptr;
                    if (!isEmpty())
                    {
                        c = m_slots[m_current_idx];
                        m_current_idx = (m_current_idx + 1) % m_capacity;
                    }
                    return c;
                }

                T* steal()
                {
                    T* c = nullptr;
                    if (!isEmpty())
                    {
                        c = m_slots[m_current_idx];
                        if (!STEAL_POLICY::canSteal(c))
                        {
                            return nullptr;
                        }
                        m_current_idx = (m_current_idx + 1) % m_capacity;
                    }
                    return c;
                }

                struct iterator
                {
                    T** data;
                };

              private:
                void resize()
                {
                    T** old_slots = m_slots;
                    m_slots = new T*[2 * m_capacity];
                    std::size_t offset = m_capacity - m_current_idx;
                    std::memcpy(m_slots, old_slots + m_current_idx, offset * sizeof(T));
                    if (0 < m_current_idx)
                    {
                        std::memcpy(m_slots + offset, old_slots, m_push_idx * sizeof(T));
                    }
                    m_current_idx = 0;
                    m_push_idx = m_capacity - 1;
                    m_capacity *= 2;
                    delete[] old_slots;
                }

                std::size_t m_push_idx;
                std::size_t m_current_idx;
                std::size_t m_capacity;
                T** m_slots;
            };

            template <class T = boost::fibers::context>
            class SpinlockQueue
            {
              private:
                mutable boost::fibers::detail::spinlock m_splk{};
                RobbableQueue<T> m_queue;

              public:
                SpinlockQueue(std::size_t capacity = 4096)
                    : m_queue{capacity}
                {
                }

                ~SpinlockQueue()
                {
                }

                SpinlockQueue(SpinlockQueue const&) = delete;
                SpinlockQueue& operator=(SpinlockQueue const&) = delete;

                bool empty() const noexcept
                {
                    boost::fibers::detail::spinlock_lock lk{m_splk};
                    return m_queue.isEmpty();
                }

                std::size_t size() const noexcept
                {
                    boost::fibers::detail::spinlock_lock lk{m_splk};
                    return m_queue.size();
                }

                void push(T* c)
                {
                    boost::fibers::detail::spinlock_lock lk{m_splk};
                    m_queue.push(c);
                }

                T* pop()
                {
                    boost::fibers::detail::spinlock_lock lk{m_splk};
                    return m_queue.pop();
                }

                T* steal()
                {
                    boost::fibers::detail::spinlock_lock lk{m_splk};
                    return m_queue.steal();
                }
            };
        } // namespace detail
    }     // namespace fibers
} // namespace mo

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_SUFFIX
#endif

#endif // BOOST_FIBERS_DETAIL_SPINLOCK_QUEUE_H
