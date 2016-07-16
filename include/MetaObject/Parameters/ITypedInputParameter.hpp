#pragma once

#include "InputParameter.hpp"
#include "ITypedParameter.hpp"

namespace mo
{
    template<class T> class ITypedInputParameter: public ITypedParameter<T>, public InputParameter
    {
    public:
        ITypedInputParameter(const std::string& name = "",  Context* ctx = nullptr);
        ~ITypedInputParameter();
        bool SetInput(std::shared_ptr<IParameter> input);
        bool SetInput(IParameter* input);

        virtual bool AcceptsInput(std::weak_ptr<IParameter> param) const;
        virtual bool AcceptsInput(IParameter* param) const;
        virtual bool AcceptsType(TypeInfo type) const;
        IParameter* GetInput();

        T* GetDataPtr(long long ts = -1, Context* ctx = nullptr);
        bool GetData(T& value, long long time_step = -1, Context* ctx = nullptr);
        T GetData(long long ts = -1, Context* ctx = nullptr);


        ITypedParameter<T>* UpdateData(T& data_, long long ts, Context* ctx){ return this;}
        ITypedParameter<T>* UpdateData(const T& data_, long long ts, Context* ctx){return this;}
        ITypedParameter<T>* UpdateData(T* data_, long long ts, Context* ctx){return this;}

    protected:
        virtual void onInputDelete(IParameter* param);
        virtual void onInputUpdate(Context* ctx, IParameter* param);
        std::shared_ptr<ITypedParameter<T>> shared_input;
        ITypedParameter<T>* input;
    private:
        std::shared_ptr<Connection> inputConnection;
        std::shared_ptr<Connection> deleteConnection;
    };
}
#include "detail/ITypedInputParameterImpl.hpp"