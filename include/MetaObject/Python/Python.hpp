#pragma once
#include "Defs.h"
#include <functional>
namespace Signals
{
	class SIGNAL_EXPORTS python_class_registry
	{
	public:
		static void setup_python_module();
		static void register_python_setup_function(const char* name, std::function<void(void)> f);
	};
}