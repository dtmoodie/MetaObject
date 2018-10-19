#include "TMultiOutput.hpp"

namespace mo
{
    IMultiOutput::IMultiOutput()
    {
        this->setFlags(mo::ParamFlags::Output_e);
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

    void IMultiOutput::visit(mo::IReadVisitor* visitor)
    {
        auto out = getOutputParam();
        if (out)
        {
            out->visit(visitor);
        }
    }

    void IMultiOutput::visit(mo::IWriteVisitor* visitor) const
    {
        auto out = getOutputParam();
        if (out)
        {
            out->visit(visitor);
        }
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
