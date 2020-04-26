#include "TParam.hpp"

namespace mo
{
#ifdef MO_TEMPLATE_EXTERN
    template MO_EXPORTS struct TParam<ISubscriber>;
    template MO_EXPORTS struct TParam<IPublisher>;
#endif // MO_TEMPLATE_EXTERN
    // template MO_EXPORTS struct TParam<IControlParam>;
} // namespace mo