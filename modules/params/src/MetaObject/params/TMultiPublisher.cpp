#include "TMultiPublisher.hpp"

namespace mo
{
    IMultiPublisher::IMultiPublisher()
    {
        this->setFlags(mo::ParamFlags::kOUTPUT);
    }

    std::vector<TypeInfo> IMultiPublisher::getOutputTypes() const
    {
        std::vector<TypeInfo> out;
        for (auto output : m_outputs)
        {
            auto tmp = output->getOutputTypes();
            out.insert(tmp.begin(), tmp.end(), out.end());
        }
        return out;
    }

    uint32_t IMultiPublisher::getNumSubscribers() const
    {
        uint32_t subs = 0;
        for (const auto& out : m_outputs)
        {
            subs += out->getNumSubscribers();
        }
        return subs;
    }

    void IMultiPublisher::setAllocator(Allocator::Ptr_t alloc)
    {
        for (auto out : m_outputs)
        {
            out->setAllocator(alloc);
        }
    }

    IPublisher* IMultiPublisher::getPublisher(const TypeInfo type)
    {
        for (auto out : m_outputs)
        {
            const auto tmp = out->getOutputTypes();
            if (std::find(tmp.begin(), tmp.end(), type) != tmp.end())
            {
                return out;
            }
        }
        return nullptr;
    }

    const IPublisher* IMultiPublisher::getPublisher(const TypeInfo type) const
    {
        for (auto out : m_outputs)
        {
            const auto tmp = out->getOutputTypes();
            if (std::find(tmp.begin(), tmp.end(), type) != tmp.end())
            {
                return out;
            }
        }
        return nullptr;
    }

    IPublisher* IMultiPublisher::getPublisher()
    {
        return getPublisher(m_current_type);
    }

    const IPublisher* IMultiPublisher::getPublisher() const
    {
        return getPublisher(m_current_type);
    }

    void IMultiPublisher::setName(std::string name)
    {
        for (auto out : m_outputs)
        {
            out->setName(name);
        }
    }

    bool IMultiPublisher::providesOutput(const TypeInfo type) const
    {
        for (auto out : m_outputs)
        {
            const auto tmp = out->getOutputTypes();
            if (std::find(tmp.begin(), tmp.end(), type) != tmp.end())
            {
                return out;
            }
        }
        return false;
    }

    std::ostream& IMultiPublisher::print(std::ostream& os) const
    {
        auto out = getPublisher();
        if (out)
        {
            out->print(os);
        }
        return os;
    }

    void IMultiPublisher::load(mo::ILoadVisitor& visitor)
    {
        auto out = getPublisher();
        if (out)
        {
            out->load(visitor);
        }
    }

    void IMultiPublisher::save(mo::ISaveVisitor& visitor) const
    {
        auto out = getPublisher();
        if (out)
        {
            out->save(visitor);
        }
    }

    void IMultiPublisher::visit(StaticVisitor& ar) const
    {
        auto out = getPublisher();
        if (out)
        {
            out->visit(ar);
        }
    }

    IDataContainerConstPtr_t IMultiPublisher::getData(const Header* desired, IAsyncStream* stream)
    {
        auto out = getPublisher();
        if (out)
        {
            return out->getData(desired, stream);
        }
        return {};
    }

    void IMultiPublisher::setOutputs(std::vector<IPublisher*>&& outputs)
    {
        this->m_outputs = std::move(outputs);
    }

    std::vector<Header> IMultiPublisher::getAvailableHeaders() const
    {
        auto out = getPublisher();
        if (out)
        {
            return out->getAvailableHeaders();
        }
        return {};
    }

    boost::optional<Header> IMultiPublisher::getNewestHeader() const
    {
        auto out = getPublisher();
        if (out)
        {
            return out->getNewestHeader();
        }
        return {};
    }
} // namespace mo
