#include "MetaObject/signals/SignalInfo.hpp"
#include <sstream>
std::string mo::SignalInfo::print()
{
    std::stringstream ss;
    ss << "- " << name << " [" << signature.name() << "]\n";
    if(tooltip.size())
        ss << "  " << tooltip << "\n";
    if(description.size())
        ss << "  " << description << "\n";
    return ss.str();
}
