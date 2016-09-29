#include "obj.hpp"

#include <MetaObject/MetaObjectFactory.hpp>
#include <boost/thread.hpp>
int main()
{
	auto factory = mo::MetaObjectFactory::Instance(); // ->RegisterTranslationUnit();
	factory->RegisterTranslationUnit();
	auto obj = rcc::shared_ptr<printable>::Create();
	
	bool recompiling = false;
	while (1)
	{
		obj->print();


		if (factory->CheckCompile())
		{
			recompiling = true;
		}
		if (recompiling)
		{
			if (factory->SwapObjects())
			{
				recompiling = false;
			}
		}
		boost::this_thread::sleep_for(boost::chrono::seconds(1));
	}

}