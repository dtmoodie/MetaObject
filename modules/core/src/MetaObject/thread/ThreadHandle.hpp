#pragma once

namespace boost
{
    namespace fiber
    {
        class scheduler;
    }
}

namespace mo
{
    class ThreadHandle
    {
      public:
      private:
        boost::fiber::scheduler* m_scheduler;
    };
}
