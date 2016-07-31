#include "MetaObject/MetaObjectInfo.hpp"
#include "MetaObject/Parameters/ParameterInfo.hpp"
#include "MetaObject/Signals/SignalInfo.hpp"
#include "MetaObject/Signals/SlotInfo.hpp"
#include <sstream>
using namespace mo;
std::string IMetaObjectInfo::Print()
{
    std::stringstream ss;
    ss << GetInterfaceId() << " ---- " << GetObjectName() << " -------\n";
    auto tooltip = GetObjectTooltip();
    if(tooltip.size())
        ss << "  " << tooltip << "\n";
    auto help = GetObjectHelp();
    if(help.size())
        ss << "    " << help << "\n";
    auto params = ListParameters();
    if(params.size())
    {
        ss << "----------- Parameters ------------- \n";
        for(auto& param : params)
        {
            ss <<  param->name;
            for(int i = 0; i < 20 - param->name.size(); ++i)
                ss << " ";
            ss<< " [" << param->data_type.name() << "]\n";
            if(param->tooltip.size())
                ss << "    " << param->tooltip << "\n";
            if(param->description.size())
                ss << "    " << param->description << "\n";
            ss << "\n";
        }
    }
    auto signals = ListSignalInfo();
    if(signals.size())
    {
        ss << "----------- Signals ---------------- \n";
        for(auto& sig : signals)
        {
            ss << sig->name;
            for(int i = 0; i < 20 - sig->name.size(); ++i)
                ss << " ";
            ss << " [" << sig->signature.name() << "]\n";
            if(sig->tooltip.size())
                ss << "    " << sig->tooltip << "\n";
            if(sig->description.size())
                ss << "    " << sig->description << "\n";
        }
    }
    
    auto slots = ListSlotInfo();
    if(slots.size())
    {
        ss << "----------- Slots ---------------- \n";
        for(auto& slot : slots)
        {
            ss << slot->name;
            for(int i = 0; i < 20 - slot->name.size(); ++i)
                ss << " ";
            ss << " [" << slot->signature.name() << "]\n";
            if(slot->tooltip.size())
                ss << "    " << slot->tooltip << "\n";
            if(slot->description.size())
                ss << "    " << slot->description << "\n";
        }
    }
    return ss.str();
}