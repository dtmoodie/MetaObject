#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Parameters/ITypedInputParameter.hpp"
#include "map.hpp"
#include "IBuffer.hpp"
namespace mo
{
    namespace Buffer
    {
        template<class T> class MO_EXPORTS StreamBuffer: public Map<T>
        {
        public:
            typedef T ValueType;

            StreamBuffer(const std::string& name = "");

            T*   GetDataPtr(long long ts = -1, Context* ctx = nullptr);
            bool GetData(T& value, long long ts = -1, Context* ctx = nullptr);
            T    GetData(long long ts = -1, Context* ctx = nullptr);
            void SetSize(long long size);
            std::shared_ptr<IParameter> DeepCopy() const;
        private:
            void prune();
            int _current_timestamp;
            int _padding;
        };
    }
}
#include "detail/StreamBufferImpl.hpp"