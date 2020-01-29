#include "obj.hpp"

int main()
{
    /*!
     * \brief factory is a pointer to the global object factory
     */

    std::shared_ptr<SystemTable> table = SystemTable::instance();
    std::shared_ptr<mo::MetaObjectFactory> factory = mo::MetaObjectFactory::instance();

    // call the inlined register translation unit function to register ConcreteImplementation
    // to the global object registry
    factory->registerTranslationUnit();

    // Get a list of objects that inherit from ExampleInterface
    auto constructors = factory->getConstructors(ExampleInterface::getHash());

    // Print static object info
    for (IObjectConstructor* constructor : constructors)
    {
        const IObjectInfo* info = constructor->GetObjectInfo();
        if (const ExampleInterfaceInfo* interface_info = dynamic_cast<const ExampleInterfaceInfo*>(info))
        {
            interface_info->PrintHelp();
        }
        // Print reflection info
        std::cout << info->Print() << std::endl;
    }

    // Construct an object
    mo::IMetaObject* obj = factory->create("ConcreteImplementation");
    if (ExampleInterface* interface_object = dynamic_cast<ExampleInterface*>(obj))
    {
        interface_object->foo();
    }
    delete obj;
    return 0;
}
