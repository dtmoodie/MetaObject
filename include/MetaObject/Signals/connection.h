#pragma once
#include "Defs.h"

#include <memory>
#include <boost/signals2/connection.hpp>
namespace Signals
{
    //typedef boost::signals2::scoped_connection connection;
    class SIGNAL_EXPORTS connection
    {
    public:
        connection(const boost::signals2::connection& connection_);
        virtual ~connection();
    private:
        boost::signals2::scoped_connection _connection;
    };

    class SIGNAL_EXPORTS class_connection: public connection
    {
        void* _connecting_class;
    public:
        class_connection(const boost::signals2::connection& connection_, void* connecting_class_);
        virtual ~class_connection();
    };
    
} // namespace Signals
