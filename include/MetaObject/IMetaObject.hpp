#pragma once
#include <IObject.h>
#include <memory>
namespace mo
{
    class Context;
    class ISignalManager;
    class ISignal;
    class Connection;
    
    class IParameter;
    struct ParameterInfo;
    struct SignalInfo;
    struct SlotInfo;
    
    class IMetaObject: public IObject
    {
    public:
        IMetaObject();
        virtual ~IMetaObject();
        
        // Setup
        virtual void SetContext(Context* ctx);
        Context* GetContext() const;
        virtual void SetupSignals(ISignalManager* mgr) = 0;

        // Introspection
        virtual std::vector<ParameterInfo*> GetParameterInfo() const = 0;
        virtual std::vector<SignalInfo*>    GetSignalInfo() const = 0;
        virtual std::vector<SlotInfo*>      GetSlotInfo() const = 0;
        std::vector<SignalInfo*>            GetSignalInfo(const std::string& name);
        std::vector<SlotInfo*>              GetSlotInfo(const std::string& name);
        

        // Signals / slots
        virtual bool ConnectByName(const std::string& name, ISignal* sig) = 0;
        virtual int  ConnectByName(const std::string& name, ISignalManager* mgr) = 0;
        virtual int  ConnectAll(ISignalManager* mgr) = 0;

        virtual int DisconnectByName(const std::string& name) = 0;
        virtual bool Disconnect(ISignal* sig);

        // Parameters
        virtual std::vector<std::weak_ptr<IParameter>> GetDisplayParameters() const = 0;
    protected:
        virtual void AddConnection(std::shared_ptr<Connection>& connection, ISignal* sig);
        Context* _ctx;

    private:
        struct impl;
        impl* _pimpl;
    };
}