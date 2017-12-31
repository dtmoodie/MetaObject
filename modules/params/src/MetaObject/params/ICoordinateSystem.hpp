#pragma once
#include "MetaObject/detail/Export.hpp"
#include <memory>
#include <string>

namespace mo
{
    class MO_EXPORTS ICoordinateSystem
    {
      public:
        ICoordinateSystem(const std::string& name);
        virtual ~ICoordinateSystem();
        virtual ICoordinateSystem* clone() const = 0;
        const std::string& getName() const;
        void setName(const std::string& name);

      private:
        std::string m_name;
    };
}
