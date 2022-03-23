

#include "MetaObject/logging/CompileLogger.hpp"
#include <MetaObject/logging/logging.hpp>
#include <spdlog/details/registry.h>
#include <stdio.h>
using namespace mo;

void CompileLogger::LogError(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    LogInternal(3, format, args);
}
void CompileLogger::LogWarning(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    LogInternal(2, format, args);
}
void CompileLogger::LogInfo(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    LogInternal(1, format, args);
}
void CompileLogger::LogDebug(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    LogInternal(0, format, args);
}

void CompileLogger::LogInternal(int severity, const char* format, va_list args)
{
    vsnprintf(m_buff, LOGSYSTEM_MAX_BUFFER - 1, format, args);
    // Make sure there's a limit to the amount of rubbish we can output
    m_buff[LOGSYSTEM_MAX_BUFFER - 1] = '\0';
    if (severity == 0)
    {
        mo::getDefaultLogger().trace(m_buff);
    }
    if (severity == 1)
    {
        mo::getDefaultLogger().debug(m_buff);
    }
    if (severity == 2)
    {
        mo::getDefaultLogger().info(m_buff);
    }
    if (severity == 3)
    {
        mo::getDefaultLogger().warn(m_buff);
    }
    if (severity == 4)
    {
        mo::getDefaultLogger().error(m_buff);
    }
}

bool BuildCallback::TestBuildCallback(const char* file, TestBuildResult type)
{
    switch (type)
    {
    case TESTBUILDRRESULT_SUCCESS: {
        mo::getDefaultLogger().info("{} {}", "TEST BUILD RESULT SUCCESS", file);
        break;
    }
    case TESTBUILDRRESULT_NO_FILES_TO_BUILD: {
        mo::getDefaultLogger().info("{} {}", "TESTBUILDRRESULT_NO_FILES_TO_BUILD", file);
        break;
    }
    case TESTBUILDRRESULT_BUILD_FILE_GONE: {
        mo::getDefaultLogger().info("{} {}", "TESTBUILDRRESULT_BUILD_FILE_GONE", file);
        break;
    }
    case TESTBUILDRRESULT_BUILD_NOT_STARTED: {
        mo::getDefaultLogger().info("{} {}", "TESTBUILDRRESULT_BUILD_NOT_STARTED", file);
        break;
    }
    case TESTBUILDRRESULT_BUILD_FAILED: {
        mo::getDefaultLogger().info("{} {}", "TESTBUILDRRESULT_BUILD_FAILED", file);
        break;
    }
    case TESTBUILDRRESULT_OBJECT_SWAP_FAIL: {
        mo::getDefaultLogger().info("{} {}", "TESTBUILDRRESULT_OBJECT_SWAP_FAIL", file);
        break;
    }
    }
    return true;
}

bool BuildCallback::TestBuildWaitAndUpdate()
{
    return true;
}
