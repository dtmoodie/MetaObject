#include "obj.hpp"

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <boost/thread.hpp>
int main()
{

    try
    {
        THROW(debug, "Test throw");
    }
    catch (std::runtime_error& e)
    {
        (void)e;
        MO_LOG(debug, "Exception caught in the correct handler");
    }
    catch (...)
    {
        MO_LOG(debug, "Exception caught in the wrong handler");
    }
    std::shared_ptr<SystemTable> table = SystemTable::instance();
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
