#pragma once

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined metaobject_metaparams_EXPORTS
#define METAPARAMTERS_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#define METAPARAMTERS_EXPORTS __attribute__((visibility("default")))
#else
#define METAPARAMTERS_EXPORTS
#endif

#ifdef _MSC_VER
#ifndef metaobject_metaparams_EXPORTS
#ifdef _DEBUG
#pragma comment(lib, "metaobject_metaparamsd.lib")
#else
#pragma comment(lib, "metaobject_metaparams.lib")
#endif
#endif
#endif

#ifdef _MSC_VER
#define META_PARAM_EXTERN(TYPE)                                                                                        \
    template class mo::TInputParamPtr<TYPE>;                                                                           \
    template class mo::TParamPtr<TYPE>;                                                                                \
    template class mo::TParamOutput<TYPE>;
#else
#define META_PARAM_EXTERN(TYPE)                                                                                        \
    extern template class mo::TInputParamPtr<TYPE>;                                                                    \
    extern template class mo::TParamPtr<TYPE>;                                                                         \
    extern template class mo::TParamOutput<TYPE>;
#endif

#include "MetaObject/core/metaobject_config.hpp"
#include <MetaObject/params/TInputParam.hpp>
#include <MetaObject/params/TParamPtr.hpp>
#include <MetaObject/types/file_types.hpp>

struct SystemTable;

namespace mo
{
    METAPARAMTERS_EXPORTS void initMetaParamsModule(SystemTable* table = nullptr);
}

META_PARAM_EXTERN(bool)
META_PARAM_EXTERN(int)
META_PARAM_EXTERN(unsigned short)
META_PARAM_EXTERN(unsigned int)
META_PARAM_EXTERN(char)
META_PARAM_EXTERN(unsigned char)
META_PARAM_EXTERN(long)
META_PARAM_EXTERN(long long)
META_PARAM_EXTERN(size_t)
META_PARAM_EXTERN(float)
META_PARAM_EXTERN(double)
META_PARAM_EXTERN(std::string)
META_PARAM_EXTERN(long)
typedef std::map<std::string, std::string> StringMap;
META_PARAM_EXTERN(StringMap);

META_PARAM_EXTERN(std::vector<int>);
META_PARAM_EXTERN(std::vector<unsigned short>);
META_PARAM_EXTERN(std::vector<unsigned int>);
META_PARAM_EXTERN(std::vector<char>);
META_PARAM_EXTERN(std::vector<unsigned char>);
META_PARAM_EXTERN(std::vector<float>);
META_PARAM_EXTERN(std::vector<double>);
META_PARAM_EXTERN(std::vector<std::string>);

#if MO_HAVE_OPENCV
#include <opencv2/core/types.hpp>
META_PARAM_EXTERN(cv::Point2f);
META_PARAM_EXTERN(cv::Point2d);
META_PARAM_EXTERN(cv::Point3d);
META_PARAM_EXTERN(cv::Point3f);
META_PARAM_EXTERN(cv::Point);

META_PARAM_EXTERN(cv::Scalar);
META_PARAM_EXTERN(cv::Vec2f);
META_PARAM_EXTERN(cv::Vec3f);
META_PARAM_EXTERN(cv::Vec2b);
META_PARAM_EXTERN(cv::Vec3b);
META_PARAM_EXTERN(std::vector<cv::Vec3b>);
typedef std::map<std::string, cv::Vec3b> ClassColormap_t;
META_PARAM_EXTERN(ClassColormap_t);

META_PARAM_EXTERN(cv::Rect);
META_PARAM_EXTERN(cv::Rect2d);
META_PARAM_EXTERN(cv::Rect2f);
META_PARAM_EXTERN(std::vector<cv::Rect>);
META_PARAM_EXTERN(std::vector<cv::Rect2f>);
#endif

META_PARAM_EXTERN(mo::ReadFile);
META_PARAM_EXTERN(mo::WriteFile);
META_PARAM_EXTERN(mo::ReadDirectory);
META_PARAM_EXTERN(mo::WriteDirectory);
META_PARAM_EXTERN(mo::EnumParam);
