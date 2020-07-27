#ifndef MO_PARAMS_FORWARD_HPP
#define MO_PARAMS_FORWARD_HPP

namespace mo
{
    // interfaces
    struct ICoordinateSystem;
    class IParam;
    struct ILoadVisitor;
    struct ISaveVisitor;
    struct IDynamicVisitor;
    class IParamServer;

    // concrete types
    struct AccessTokenLock;
    class ISubscriber;

    class ParamFactory;

    // templated types
    template <class T>
    struct AccessToken;
} // namespace mo

#endif // MO_PARAMS_FORWARD_HPP
