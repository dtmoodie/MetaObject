#include "TMultiOutput.hpp"

namespace mo
{
    IMultiOutput::IMultiOutput()
    {
        this->setFlags(mo::ParamFlags::kOUTPUT);
    }

    std::vector<TypeInfo> IMultiOutput::listOutputTypes() const
    {
        std::vector<TypeInfo> out;
        for (auto output : m_outputs)
        {
            out.push_back(output->getTypeInfo());
        }
        return out;
    }

    ParamBase* IMultiOutput::getOutputParam(const TypeInfo type)
    {
        for (auto out : m_outputs)
        {
            if (out->getTypeInfo() == type)
            {
                return out;
            }
        }
        return nullptr;
    }

    const ParamBase* IMultiOutput::getOutputParam(const TypeInfo type) const
    {
        for (auto out : m_outputs)
        {
            if (out->getTypeInfo() == type)
            {
                return out;
            }
        }
        return nullptr;
    }

    ParamBase* IMultiOutput::getOutputParam()
    {
        return getOutputParam(m_current_type);
    }

    const ParamBase* IMultiOutput::getOutputParam() const
    {
        return getOutputParam(m_current_type);
    }

    void IMultiOutput::setName(const std::string& name)
    {
        for (auto out : m_outputs)
        {
            out->setName(name);
        }
    }

    bool IMultiOutput::providesOutput(const TypeInfo type) const
    {
        for (auto out : m_outputs)
        {
            if (out->getTypeInfo() == type)
            {
                return out;
            }
        }
        return false;
    }

    TypeInfo IMultiOutput::getTypeInfo() const
    {
        return m_current_type;
    }

    std::ostream& IMultiOutput::print(std::ostream& os) const
    {
        auto out = getOutputParam();
        if (out)
        {
            out->print(os);
        }
        return os;
    }

    void IMultiOutput::load(mo::ILoadVisitor& visitor)
    {
        auto out = getOutputParam();
        if (out)
        {
            out->load(visitor);
        }
    }

    void IMultiOutput::save(mo::ISaveVisitor& visitor) const
    {
        auto out = getOutputParam();
        if (out)
        {
            out->save(visitor);
        }
    }

    void IMultiOutput::load(BinaryInputVisitor& ar)
    {
        auto out = getOutputParam();
        if (out)
        {
            out->load(ar);
        }
    }

    void IMultiOutput::save(BinaryOutputVisitor& ar) const
    {
        auto out = getOutputParam();
        if (out)
        {
            out->save(ar);
        }
    }

    void IMultiOutput::visit(StaticVisitor& ar) const
    {
        auto out = getOutputParam();
        if (out)
        {
            out->visit(ar);
        }
    }

    OptionalTime IMultiOutput::getTimestamp() const
    {
        auto out = getOutputParam();
        if (out)
        {
            return out->getTimestamp();
        }
        return {};
    }

    FrameNumber IMultiOutput::getFrameNumber() const
    {
        auto out = getOutputParam();
        if (out)
        {
            return out->getFrameNumber();
        }
        return {};
    }

    IMultiOutput::IContainerPtr_t IMultiOutput::getData(const Header& header)
    {
        auto out = getOutputParam();
        if (out)
        {
            return out->getData(header);
        }
        return {};
    }

    IMultiOutput::IContainerConstPtr_t IMultiOutput::getData(const Header& header) const
    {
        auto out = getOutputParam();
        if (out)
        {
            return out->getData(header);
        }
        return {};
    }

    void IMultiOutput::setOutputs(const std::vector<IParam*>& outputs)
    {
        this->m_outputs = outputs;
    }
}
