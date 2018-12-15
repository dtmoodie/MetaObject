#pragma once
#include <memory>
#include <list>
#include <functional>
#include <mutex>

namespace mo
{
    template<class T>
    struct ObjectPool
    {
        using Ptr = std::shared_ptr<T>;

        ObjectPool(std::function<Ptr()>&& constructor = []()->Ptr{return std::make_shared<T>();}):
            m_constructor(std::move(constructor))
        {

        }

        Ptr getObject()
        {
            Ptr owning;
            {
                std::lock_guard<std::mutex> lock(m_mtx);

                if(m_objects.empty())
                {
                    owning = m_constructor();
                }else
                {
                    owning = m_objects.front();
                    m_objects.pop_front();
                }
            }

            Ptr non_owning(owning.get(), [owning, this](T*)
                {
                    returnObject(std::move(owning));
                });
            return non_owning;
        }

    private:


        void returnObject(const std::shared_ptr<T>& obj)
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_objects.push_back(std::move(obj));
        }

        std::list<Ptr> m_objects;
        std::function<Ptr()> m_constructor;
        std::mutex m_mtx;
    };
}
