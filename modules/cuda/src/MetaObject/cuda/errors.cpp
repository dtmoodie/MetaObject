#include <MetaObject/cuda/errors.hpp>

namespace std
{
    ostream& operator<<(ostream& os, const std::vector<const char*>& vec)
    {
        if (!vec.empty())
        {
            os << "[";
            for (size_t i = 0; i < vec.size(); ++i)
            {
                if (i != 0)
                {
                    os << " ";
                }
                os << vec[i];
            }
            os << "]";
        }
        return os;
    }
} // namespace std
