#include "ICoordinateSystem.hpp"

namespace mo
{
    ICoordinateSystem::ICoordinateSystem(const std::string& name) : m_name(name) {}

    ICoordinateSystem::~ICoordinateSystem() {}

    const std::string& ICoordinateSystem::getName() const { return m_name; }

    void ICoordinateSystem::setName(const std::string& name) { m_name = name; }
}
