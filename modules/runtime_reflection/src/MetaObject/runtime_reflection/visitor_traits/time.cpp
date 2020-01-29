#include "time.hpp"

namespace mo
{
    void TTraits<Time, 4>::load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const
    {
        auto& ref = this->ref(inst);
        double sec;
        visitor(&sec, "seconds");
        ref = Time(sec);
    }

    void TTraits<Time, 4>::save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t) const
    {
        const auto& ref = this->ref(inst);
        const auto sec = ref.seconds();
        visitor(&sec, "seconds");
    }

    void TTraits<Time, 4>::visit(StaticVisitor& visitor, const std::string&) const
    {
        visitor.template visit<double>("seconds");
    }
}
