#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"

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
				map = 2
			};
			typedef std::function<IParameter*(IParameter*)> create_buffer_f;

			void RegisterFunction(TypeInfo type, const create_buffer_f& func, buffer_type buffer_type_);
			std::shared_ptr<IParameter> CreateProxy(std::weak_ptr<IParameter> param, buffer_type buffer_type_);

			static BufferFactory* Instance();
		private:
			std::map<Loki::TypeInfo, std::map<buffer_type, create_buffer_f>> _registered_buffer_factories;
		};
	}
}