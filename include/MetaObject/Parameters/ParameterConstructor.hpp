#include "ParameterFactory.hpp"
namespace mo
{
	template<class T1, class T2, int Type> class ParameterConstructor
	{
	public:
		ParameterConstructor()
		{
			ParameterFactory::instance()->RegisterConstructor(TypeInfo(typeid(T2)),
				std::bind(&ParameterConstructor<T1, T2, Type>::create), Type);

			ParameterFactory::instance()->RegisterConstructor(TypeInfo(typeid(T1)),
				std::bind(&ParameterConstructor<T1, T2, Type>::create));
		}
		static IParameter* create()
		{
			return new T1();
		}
	};
}