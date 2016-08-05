#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <functional>
#include <memory>
namespace mo
{
	class IParameter;


	namespace Buffer
	{
		class MO_EXPORTS BufferFactory
		{
		public:
			enum buffer_type
			{
				cbuffer = 0,
				cmap = 1,
				map = 2,
                StreamBuffer = 3
			};
			typedef std::function<IParameter*(IParameter*)> create_buffer_f;

			static void RegisterFunction(TypeInfo type, const create_buffer_f& func, buffer_type buffer_type_);
			static std::shared_ptr<IParameter> CreateProxy(IParameter* param, buffer_type buffer_type_);
		};
	}
}