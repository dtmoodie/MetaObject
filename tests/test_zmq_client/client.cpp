#ifdef HAVE_ZEROMQ


#define BOOST_TEST_MAIN

#include "MetaObject/object/IMetaObject.hpp"
#include "MetaObject/object/detail/IMetaObjectImpl.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParam.hpp"
#include "MetaObject/params/IO/SerializationFactory.hpp"
#include "MetaObject/params/ParamClient.hpp"
#include "cereal/archives/portable_binary.hpp"
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"


#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif
#include <iostream>
#include "zmq.hpp"
#include "zmq_utils.h"
using namespace mo;


BOOST_AUTO_TEST_CASE(client)
{
    auto inst = ParamClient::Instance();
    inst->connect("tcp://localhost:5566");
    


    /*zmq::context_t ctx(1);

    zmq::socket_t socket(ctx, ZMQ_SUB);
    socket.connect("tcp://localhost:5566");
    const char* topic = "update_topic";
    socket.setsockopt(ZMQ_SUBSCRIBE, topic, strlen(topic));
    zmq::message_t msg;
    TParam<int> Param;
    Param.updateData(0);
    auto deserialization_func = SerializationFactory::instance()->GetBinaryDeSerializationFunction(Param.getTypeInfo());
    int count = 0;
    while(1)
    {
        if(socket.recv(&msg))
        {
            std::istringstream iss(static_cast<char*>(msg.data()));
            cereal::BinaryInputArchive ar(iss);
            deserialization_func(&Param, ar);
            mo::Time_t ts = Param.getTimestamp();
            int value = Param.GetData();
            BOOST_REQUIRE_EQUAL(ts, value);
            ++count;
            if(count > 100)
                break;
        }
    }*/
}





#else
#include <iostream>
int main()
{
    std::cout << "Not build with zero mq supprt";
    return 0;
}
#endif
