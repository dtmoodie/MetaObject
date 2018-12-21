#include <MetaObject/visitation.hpp>

void staticCheckFunction()
{
    static_assert(mo::is_complete<mo::TTraits<std::string>>::value, "mo::is_complete<mo::TTraits<std::string>>::value");
}
