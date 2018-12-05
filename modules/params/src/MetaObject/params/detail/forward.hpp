#ifndef MO_PARAMS_FORWARD_HPP
#define MO_PARAMS_FORWARD_HPP

namespace mo
{
    // interfaces
    class ICoordinateSystem;
    class IParam;
    struct IReadVisitor;
    struct IWriteVisitor;
    struct IDynamicVisitor;
    class IParamServer;

    // concrete types
    struct AccessTokenLock;
    class InputParamAny;
    class InputParam;

    class ParamFactory;

    // templated types
    template <class T>
    struct AccessToken;
}

#endif // MO_PARAMS_FORWARD_HPP
