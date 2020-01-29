#pragma once
#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION

#include "converters.hpp"
#include "numpy/ndarraytypes.h"
#include <MetaObject/Python.hpp>
#include <MetaObject/core/detail/Allocator.hpp>
#include <Python.h>

#include <ct/interop/boost_python/PythonConverter.hpp>

#include <opencv2/core/mat.hpp>

namespace ct
{
    template <>
    struct MO_EXPORTS PythonConverter<cv::Mat, 5, void>
    {
        static boost::python::object convertToPython(const cv::Mat& mat);
        static bool convertFromPython(const boost::python::object& obj, cv::Mat& result);
        static void registerToPython(const char* name);
    };

    class PyEnsureGIL
    {
      public:
        PyEnsureGIL();
        ~PyEnsureGIL();

      private:
        PyGILState_STATE _state;
    };
}
namespace mo
{
    void setupAllocator();
    class MO_EXPORTS NumpyAllocator : virtual public cv::MatAllocator
    {
      public:
        NumpyAllocator(cv::MatAllocator* default_allocator_ = nullptr);
        ~NumpyAllocator() override;

        cv::Mat fromPython(PyObject* arr) const;

        PyObject* toPython(const cv::Mat& mat) const;

        virtual cv::UMatData* allocate(int dims0,
                                       const int* sizes,
                                       int type,
                                       void* data,
                                       size_t* step,
                                       int flags,
                                       cv::UMatUsageFlags usageFlags) const override;

        virtual bool allocate(cv::UMatData* u, int accessFlags, cv::UMatUsageFlags usageFlags) const override;

        void deallocate(cv::UMatData* u) const override;

        cv::MatAllocator* default_allocator;
    };
}
