#include "obj.hpp"

#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/logging/Log.hpp>
#include <boost/thread.hpp>
int main()
{

    try
    {
        THROW(debug) << "Test throw";

    }catch(mo::ExceptionWithCallStack<std::string>& e)
    {
        LOG(debug) << "Exception caught in the correct handler";
    }catch(...)
    {
        LOG(debug) << "Exception caught in the wrong handler";
    }

    auto factory = mo::MetaObjectFactory::instance(); // ->registerTranslationUnit();
    factory->registerTranslationUnit();
    auto obj = rcc::shared_ptr<printable>::create();

    bool recompiling = false;
    while (1)
    {
        obj->print();


        if (factory->checkCompile())
        {
            recompiling = true;
        }
        if (recompiling)
        {
            if (factory->swapObjects())
            {
                recompiling = false;
            }
        }
        boost::this_thread::sleep_for(boost::chrono::seconds(1));
    }

}
