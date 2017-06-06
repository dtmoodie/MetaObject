#pragma once
#include "MetaObject/detail/Export.hpp"
#include <string>
namespace mo{
    class MO_EXPORTS ICoordinateSystem{
    public:
        ICoordinateSystem(const std::string& name):m_name(name){}

        inline const std::string& getName() const{return m_name;}
        inline void setName(const std::string& name){m_name = name;}
    private:
        std::string m_name;
    };
}
