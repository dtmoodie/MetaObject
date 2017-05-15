#ifdef HAVE_ZEROMQ


#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/detail/IMetaObjectImpl.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/detail/Counter.hpp"
#include "MetaObject/detail/MetaObjectMacros.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TParamPtr.hpp"
#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParam.hpp"
#include "MetaObject/params/IO/SerializationFunctionRegistry.hpp"
#include "MetaObject/params/IO/Policy.hpp"
#include "MetaObject/params/VariableManager.h"
#include "MetaObject/params/ParamServer.hpp"
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
#include <sstream>
#include "zmq.hpp"
#include "zmq_utils.h"
using namespace mo;


BOOST_AUTO_TEST_CASE(server)
{
    VariableManager mgr;
    TParam<int> param("test");
    mgr.AddParam(&param);
    auto inst = ParamServer::Instance();
    inst->Bind("tcp://*:5566");
    inst->Publish(&mgr, ":test");
    std::this_thread::sleep_for(std::chrono::seconds(100));
    /*zmq::context_t ctx(1);

    zmq::socket_t socket(ctx, ZMQ_PUB);
    socket.bind("tcp://*:5566");
    std::string topic_name = "update_topic";
    zmq::message_t topic(topic_name.size());

    TParam<int> Param;
    Param.UpdateData(0);
    auto serialize_func = SerializationFunctionRegistry::Instance()->GetBinarySerializationFunction(Param.GetTypeInfo());
    BOOST_REQUIRE(serialize_func);
    int count = 0;
    while(1)
    {
        Param.UpdateData(count, count);
        socket.send(topic, ZMQ_SNDMORE);
        std::stringstream oss;
        {
            cereal::BinaryOutputArchive ar(oss);
            serialize_func(&Param, ar);
        }
        std::string msg = oss.str();
        zmq::message_t msg_(msg.c_str(), msg.size());
        socket.send(msg_);
        ++count;
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
