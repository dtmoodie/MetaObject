#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.hpp"
namespace mo
{
	class ISlot;
	class ISignal;
	class Connection;
	template<class Sig> class TSlot;
	template<class Sig> class TSignal;
	class MO_EXPORTS ISignalRelay
	{
	public:
		virtual ~ISignalRelay() {}
		virtual TypeInfo getSignature() const = 0;
        virtual bool HasSlots() const = 0;
	protected:
		friend class ISlot;
		friend class ISignal;
		template<class T> friend class TSignal;
		template<class T> friend class TSlot;
		virtual bool connect(ISlot* slot) = 0;
		virtual bool disconnect(ISlot* slot) = 0;

		virtual bool connect(ISignal* signal) = 0;
		virtual bool disconnect(ISignal* signal) = 0;
	};
}
