#pragma once
#include "MetaObject/detail/Export.hpp"
#include <memory>
#include <string>

namespace mo
{
    struct MO_EXPORTS ICoordinateSystem
    {
        ICoordinateSystem(const std::string& name);
        virtual ~ICoordinateSystem();
        virtual ICoordinateSystem* clone() const = 0;
        const std::string& getName() const;
        void setName(const std::string& name);

      private:
        std::string m_name;
    };
}
