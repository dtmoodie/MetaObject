#pragma once
#include "MetaObject/detail/Export.hpp"
#include "RuntimeCompiler/ICompilerLogger.h"
#include "RuntimeObjectSystem/IRuntimeObjectSystem.h"
#include <cstdarg>
namespace mo
{
    class MO_EXPORTS CompileLogger : public ICompilerLogger
    {
      public:
        void LogError(const char* format, ...) override;
        void LogWarning(const char* format, ...) override;
        void LogInfo(const char* format, ...) override;
        void LogDebug(const char* format, ...) override;

      protected:
        void LogInternal(int severity, const char* format, va_list args);
        static const size_t LOGSYSTEM_MAX_BUFFER = 409600;
        char m_buff[LOGSYSTEM_MAX_BUFFER];
    };

    class MO_EXPORTS BuildCallback : public ITestBuildNotifier
    {
        bool TestBuildCallback(const char* file, TestBuildResult type) override;
        bool TestBuildWaitAndUpdate() override;
    };
} // namespace mo
