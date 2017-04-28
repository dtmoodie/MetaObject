#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/Enums.hpp"
#include "MetaObject/Detail/TypeInfo.hpp"
#include <functional>
#include <memory>
namespace mo
{
	class IParam;


	namespace Buffer
	{
		class MO_EXPORTS BufferFactory
		{
		public:
			typedef std::function<IParam*(IParam*)> create_buffer_f;

			static void RegisterFunction(TypeInfo type, const create_buffer_f& func, ParamType buffer_type_);
			static std::shared_ptr<IParam> CreateProxy(IParam* param, ParamType buffer_type_);
		};
	}
}