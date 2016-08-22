#ifdef HAVE_ZEROMQ

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Detail/IMetaObjectImpl.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Parameters/ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/Parameters/TypedParameter.hpp"
#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"
#include "MetaObject/Parameters/IO/Policy.hpp"
#include "cereal/archives/portable_binary.hpp"
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"


#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "parameter"
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <sstream>
#include "zmq.hpp"
#include "zmq_utils.h"
using namespace mo;


BOOST_AUTO_TEST_CASE(server)
{
    zmq::context_t ctx(1);

    zmq::socket_t socket(ctx, ZMQ_PUB);
    socket.bind("tcp://*:5566");
    std::string topic_name = "update_topic";
    zmq::message_t topic(topic_name.size());

    TypedParameter<int> parameter;
    parameter.UpdateData(0);
    auto serialize_func = SerializationFunctionRegistry::Instance()->GetBinarySerializationFunction(parameter.GetTypeInfo());
    BOOST_REQUIRE(serialize_func);
    int count = 0;
    while(1)
    {
        parameter.UpdateData(count, count);
        socket.send(topic, ZMQ_SNDMORE);
        std::stringstream oss;
        {
            cereal::PortableBinaryOutputArchive ar(oss);
            serialize_func(&parameter, ar);
        }
        std::string msg = oss.str();
        zmq::message_t msg_(msg.c_str(), msg.size());
        socket.send(msg_);
        ++count;
    }
}





#else
#include <iostream>
int main()
{
    std::cout << "Not build with zero mq supprt";
    return 0;
}
#endif