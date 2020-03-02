#ifndef MO_SIGNALS_CONNECTION_HPP
#define MO_SIGNALS_CONNECTION_HPP
#include "MetaObject/detail/Export.hpp"
#include <memory>
#include <vector>
namespace mo
{
    class ISlot;
    class ISignal;
    class ISignalRelay;
    class IMetaObject;

    // A Connection is only valid for as long as the underlying slot is valid
    class MO_EXPORTS Connection
    {
      public:
        using Ptr_t = std::shared_ptr<Connection>;

        Connection() = default;
        Connection(const Connection&) = delete;
        Connection(Connection&&) = delete;
        Connection& operator=(const Connection&) = delete;
        Connection& operator=(Connection&&) = delete;

        virtual ~Connection();
        virtual bool disconnect() = 0;
    };

    class MO_EXPORTS SlotConnection : public Connection
    {
      public:
        SlotConnection(ISlot* slot, const std::shared_ptr<ISignalRelay>& relay);

        SlotConnection(const SlotConnection&) = delete;
        SlotConnection(SlotConnection&&) = delete;
        SlotConnection& operator=(const SlotConnection&) = delete;
        SlotConnection& operator=(SlotConnection&&) = delete;

        ~SlotConnection() override;
        bool disconnect() override;

      protected:
        ISlot* _slot;
        std::weak_ptr<ISignalRelay> _relay;
    };

    class MO_EXPORTS ClassConnection : public SlotConnection
    {
      public:
        ClassConnection(const ClassConnection&) = delete;
        ClassConnection(ClassConnection&&) = delete;
        ClassConnection& operator=(const ClassConnection&) = delete;
        ClassConnection& operator=(ClassConnection&&) = delete;

        ClassConnection(ISlot* slot, std::shared_ptr<ISignalRelay> relay, IMetaObject* obj);
        ~ClassConnection() override;

        bool disconnect() override;

      protected:
        IMetaObject* _obj;
    };

    class MO_EXPORTS SignalConnection : public Connection
    {
      public:
        SignalConnection(const SignalConnection&) = delete;
        SignalConnection(SignalConnection&&) = delete;
        SignalConnection& operator=(const SignalConnection&) = delete;
        SignalConnection& operator=(SignalConnection&&) = delete;

        SignalConnection(ISignal* signal, const std::shared_ptr<ISignalRelay>& relay);

        ~SignalConnection() override = default;
        bool disconnect() override;

      protected:
        ISignal* _signal;
        std::weak_ptr<ISignalRelay> _relay;
    };

    class MO_EXPORTS ConnectionSet : public Connection
    {
      public:
        ConnectionSet(std::vector<Connection::Ptr_t>&& connections);
        bool disconnect() override;

      private:
        std::vector<Connection::Ptr_t> m_connections;
    };

} // namespace mo
#endif // MO_SIGNALS_CONNECTION_HPP