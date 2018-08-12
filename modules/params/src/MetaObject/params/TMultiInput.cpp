#include "TMultiInput.hpp"

namespace mo
{
MultiConnection::MultiConnection(std::vector<std::shared_ptr<Connection>>&& connections):
    m_connections(connections)
{

}

MultiConnection::~MultiConnection()
{

}


bool MultiConnection::disconnect()
{
    for(const auto& con : m_connections)
    {
        if(con)
        {
            con->disconnect();
        }
    }
    return true;
}
}
