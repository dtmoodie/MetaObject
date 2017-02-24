#pragma once

#include "InputParameter.hpp"
#include "ITypedParameter.hpp"
#ifdef _MSC_VER
#pragma warning( disable : 4250)
#endif
namespace mo
{
    template<class T>
class ITypedInputParameter: virtual public ITypedParameter<T>, virtual public InputParameter
    {
    public:
        ITypedInputParameter(const std::string& name = "",  Context* ctx = nullptr);
        ~ITypedInputParameter();
        bool SetInput(std::shared_ptr<IParameter> input);
        bool SetInput(IParameter* input);

        virtual bool AcceptsInput(std::weak_ptr<IParameter> param) const;
        virtual bool AcceptsInput(IParameter* param) const;
        virtual bool AcceptsType(TypeInfo type) const;

        IParameter* GetInputParam();
        
        bool GetInput(mo::time_t ts);
        bool GetInput(size_t fn);

        T*   GetDataPtr(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t* fn_ = nullptr);
        T*   GetDataPtr(size_t fn, Context* ctx = nullptr, mo::time_t* ts_ = nullptr);

        T    GetData(mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t* fn = nullptr);
        T    GetData(size_t fn, Context* ctx = nullptr, mo::time_t* ts = nullptr);

        bool GetData(T& value, mo::time_t ts = -1 * mo::second, Context* ctx = nullptr, size_t* fn = nullptr);
        bool GetData(T& value, size_t fn, Context* ctx = nullptr, mo::time_t* ts = nullptr);

        ITypedParameter<T>* UpdateData(const T& data,
                                       mo::time_t ts = -1 * mo::second,
                                       Context* ctx = nullptr,
                                       size_t fn = std::numeric_limits<size_t>::max(),
                                       ICoordinateSystem* cs = nullptr){}

    protected:
        virtual void onInputDelete(IParameter const* param);
        virtual void onInputUpdate(Context* ctx, IParameter* param);
        std::shared_ptr<ITypedParameter<T>> shared_input;
        ITypedParameter<T>* input;

    private:
		TypedSlot<void(Context*, IParameter*)> update_slot;
		TypedSlot<void(IParameter const*)> delete_slot;
    };
}
#include "detail/ITypedInputParameterImpl.hpp"
