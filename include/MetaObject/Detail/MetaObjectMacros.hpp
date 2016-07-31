#pragma once
#include <boost/preprocessor.hpp>
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObjectMacrosImpl.hpp"
#include "MetaObject/MetaObjectInfo.hpp"

// -------------------------------------------------------------------------------------------
#ifdef _MSC_VER
#define MO_BEGIN(...) BOOST_PP_CAT(BOOST_PP_OVERLOAD(MO_BEGIN_, __VA_ARGS__)(__VA_ARGS__, __COUNTER__), BOOST_PP_EMPTY())
#else
#define MO_BEGIN(...) BOOST_PP_OVERLOAD(MO_BEGIN_, __VA_ARGS__)(__VA_ARGS__, __COUNTER__)
#endif

namespace mo
{
	template<class T>
	struct Void {
		typedef void type;
	};

	template<class T, class U = void>
	struct has_parent {
		enum { value = 0 };
	};

	template<class T>
	struct has_parent<T, typename Void<typename T::PARENT_CLASS>::type > {
		enum { value = 1 };
	};
}


#define MO_END MO_END_(__COUNTER__)




