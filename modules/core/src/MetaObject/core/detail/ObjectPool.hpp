#ifndef MO_CORE_OBJECT_POOL_HPP
#define MO_CORE_OBJECT_POOL_HPP
#include "ObjectConstructor.hpp"

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/thread/fiber_include.hpp>

#include <list>

namespace mo
{
    template <class T>
    struct ObjectPool
    {
        using Ptr_t = std::shared_ptr<T>;

        ObjectPool(const uint64_t initial_pool_size, const ObjectConstructor<T>& ctr = ObjectConstructor<T>())
            : m_ctr(ctr)
        {
            for (uint64_t i = 0; i < initial_pool_size; ++i)
            {
                Ptr_t obj = m_ctr.createShared();
                returnObject(obj);
            }
        }

        ObjectPool(const ObjectConstructor<T>& ctr = ObjectConstructor<T>())
            : m_ctr(ctr)
        {
        }

        Ptr_t get()
        {
            std::lock_guard<boost::fibers::mutex> lock(m_mtx);
            Ptr_t owning_ptr;
            if (m_free_objects.empty())
            {
                owning_ptr = m_ctr.createShared();
            }
            else
            {
                owning_ptr = m_free_objects.front();
                m_free_objects.pop_front();
            }
            MO_ASSERT(owning_ptr);
            return Ptr_t(owning_ptr.get(), [this, owning_ptr](T*) { returnObject(owning_ptr); });
        }

      private:
        void returnObject(const Ptr_t& obj)
        {
            std::lock_guard<boost::fibers::mutex> lock(m_mtx);
            m_free_objects.push_front(std::move(obj));
        }

        std::list<Ptr_t> m_free_objects;
        ObjectConstructor<T> m_ctr;
        boost::fibers::mutex m_mtx;
    };
}

#endif // MO_CORE_OBJECT_POOL_HPP
