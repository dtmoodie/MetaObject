#include "MetaObject/Signals/Connection.hpp"
#include "MetaObject/Thread/InterThread.hpp"


using namespace mo;

Connection::Connection(const boost::signals2::connection& connection_):
    _connection(connection_)
{

}
Connection::~Connection()
{

}

ClassConnection::ClassConnection(const boost::signals2::connection& connection_, void* connecting_class_):
    Connection(connection_), _connecting_class(connecting_class_)
{

}
ClassConnection::~ClassConnection()
{
    ThreadSpecificQueue::RemoveFromQueue(_connecting_class);
}
