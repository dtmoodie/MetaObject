#include "MetaObject/object/IMetaObjectInfo.hpp"
#include "MetaObject/params/ParamInfo.hpp"
#include "MetaObject/core/Demangle.hpp"
#include "MetaObject/signals/SignalInfo.hpp"
#include "MetaObject/signals/SlotInfo.hpp"
#include "MetaObjectFactory.hpp"
#include "RuntimeObjectSystem/ObjectInterface.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
#include <RuntimeObjectSystem/IRuntimeObjectSystem.h>
#include <RuntimeCompiler/FileSystemUtils.h>
#include <sstream>
using namespace mo;

std::string IMetaObjectInfo::getObjectTooltip() const{
    return "";
}
std::string IMetaObjectInfo::getObjectHelp() const{
    return "";
}
std::string IMetaObjectInfo::Print(Verbosity verbosity) const
{
    std::stringstream ss;
    ss << "\n\n";
    std::string name = GetObjectName();
    ss << GetInterfaceName();
    //ss << getInterfaceId(); // << " *** " << getObjectName() << " ***\n";
    ss << " ***** ";
    ss << name << " ";
    if(name.size() < 20)
        for(int i = 0; i < 20 - static_cast<int>(name.size()); ++i)
            ss << "*";
    ss << "\n";
    IObjectConstructor* ctr = GetConstructor();
    if(verbosity >= DEBUG){
        ss << ctr->GetProjectId();
        ss << " - ";
        ss << ctr->GetPerModuleInterface()->GetModuleFileName() << "\n";
        ss << ctr->GetFileName() << "\n";
    }
    if(verbosity >= RCC){
        ss << "Num constructed objects: " << ctr->GetNumberConstructedObjects() << '\n';
        unsigned short pid = ctr->GetProjectId();
        ss << "\nIncludes:\n";
        for(size_t i = 0; i < ctr->GetMaxNumIncludeFiles(); ++i){
            const char* file = ctr->GetIncludeFile(i);
            if(file)
                ss << file << '\n';
        }
        std::vector<FileSystemUtils::Path>& inlcude_paths = MetaObjectFactory::instance()->getObjectSystem()->GetIncludeDirList(pid);
        if(inlcude_paths.size()){
            for(FileSystemUtils::Path& path : inlcude_paths){
              ss << path.m_string << '\n';
            }
        }

        ss << "\nLink libs:\n";
        for(size_t i = 0; i < ctr->GetMaxNumLinkLibraries(); ++i){
            const char* lib = ctr->GetLinkLibrary(i);
            if(lib)
                ss << lib << '\n';
        }
        ss << "\nSource dependencies:\n";
        for(size_t i = 0; i < ctr->GetMaxNumSourceDependencies(); ++i){
            SourceDependencyInfo info = ctr->GetSourceDependency(i);
            if(info.filename)
                ss << info.filename << '\n';
        }


        std::vector<FileSystemUtils::Path>& link_paths = MetaObjectFactory::instance()->getObjectSystem()->GetLinkDirList(pid);
        if(link_paths.size()){
            ss << "\nLink dirs:\n";
            for(FileSystemUtils::Path& path : link_paths){
              ss << path.m_string << '\n';
            }
        }
        ss << "\n";
    }

    auto tooltip = getObjectTooltip();
    if(tooltip.size())
        ss << "  " << tooltip << "\n";
    auto help = getObjectHelp();
    if(help.size())
        ss << "    " << help << "\n";
    auto params = getParamInfo();
    if(params.size())
    {
        ss << "----------- Params ------------- \n";
        int longest_name = 0;
        for(auto& slot : params)
            longest_name = std::max<int>(longest_name, static_cast<int>(slot->name.size()));
        longest_name += 1;
        for(auto& param : params)
        {
            ss <<  param->name;
            for(int i = 0; i < longest_name - static_cast<int>(param->name.size()); ++i)
                ss << " ";
            if(param->type_flags & Control_e)
                ss << "C";
            if(param->type_flags & Input_e)
                ss << "I";
            if(param->type_flags & Output_e)
                ss << "O";

            ss << " [" << mo::Demangle::TypeToName(param->data_type) << "]\n";
            if(param->tooltip.size())
                ss << "    " << param->tooltip << "\n";
            if(param->description.size())
                ss << "    " << param->description << "\n";
            if(param->initial_value.size())
                ss << "    " << param->initial_value << "\n";
        }
    }
    auto sigs = getSignalInfo();
    if(sigs.size())
    {
        ss << "\n----------- Signals ---------------- \n";
        int longest_name = 0;
        for(auto& slot : sigs)
            longest_name = std::max<int>(longest_name, static_cast<int>(slot->name.size()));
        longest_name += 1;
        for(auto& sig : sigs)
        {
            ss << sig->name;
            for(int i = 0; i < longest_name - static_cast<int>(sig->name.size()); ++i)
                ss << " ";
            ss << " [" << sig->signature.name() << "]\n";
            if(sig->tooltip.size())
                ss << "    " << sig->tooltip << "\n";
            if(sig->description.size())
                ss << "    " << sig->description << "\n";
        }
    }

    auto my_slots = getSlotInfo();
    if(my_slots .size())
    {
        ss << "\n----------- Slots ---------------- \n";
        int longest_name = 0;
        for(auto& slot : my_slots)
            longest_name = std::max<int>(longest_name, static_cast<int>(slot->name.size()));
        longest_name += 1;
        for(auto& slot : my_slots)
        {
            ss << slot->name;
            for(int i = 0; i < longest_name - static_cast<int>(slot->name.size()); ++i)
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
