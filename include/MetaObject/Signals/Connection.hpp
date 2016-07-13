#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <memory>
#include <boost/signals2/connection.hpp>
namespace mo
{
    class MO_EXPORTS Connection
    {
    public:
        Connection(const boost::signals2::connection& connection_);
        virtual ~Connection();
    private:
        boost::signals2::scoped_connection _connection;
    };

    class MO_EXPORTS ClassConnection: public Connection
    {
        void* _connecting_class;
    public:
        ClassConnection(const boost::signals2::connection& connection_, void* connecting_class_);
        virtual ~ClassConnection();
    };
    
} // namespace Signals
