#pragma once
#include "converters.hpp"
#include "numpy/ndarraytypes.h"
#include <MetaObject/Python.hpp>
#include <MetaObject/core/detail/Allocator.hpp>
#include <MetaObject/core/detail/opencv_allocator.hpp>
#include <Python.h>
#include <opencv2/core/mat.hpp>

namespace mo
{
    namespace python
    {
        template <>
        MO_EXPORTS boost::python::object convertToPython(const cv::Mat& mat);

        template <>
        MO_EXPORTS void convertFromPython(const boost::python::object& obj, cv::Mat& result);

        class PyEnsureGIL
        {
          public:
            PyEnsureGIL();
            ~PyEnsureGIL();
          private:
            PyGILState_STATE _state;
        };
    }

    void setupAllocator();
    class MO_EXPORTS NumpyAllocator : virtual public cv::MatAllocator
    {
      public:
        NumpyAllocator(cv::MatAllocator* default_allocator_ = nullptr);
        ~NumpyAllocator();

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
