#include "PythonAllocator.hpp"
#include "MetaObject/logging/logging.hpp"

#include <numpy/ndarrayobject.h>

#include <boost/python.hpp>

namespace mo
{
    namespace python
    {
        template <>
        boost::python::object convertToPython(const cv::Mat& mat)
        {
            if (!mat.empty())
            {
                const NumpyAllocator* alloc = dynamic_cast<const NumpyAllocator*>(mat.u->currAllocator);
                if (alloc)
                {
                    auto arr = alloc->toPython(mat);
                    if (arr)
                    {
                        return boost::python::object(boost::python::handle<>(arr));
                    }
                }
            }
            return {};
        }
    }
    struct NumpyDeallocator
    {
        NumpyDeallocator(cv::UMatData* data_, size_t size_, const NumpyAllocator* allocator_)
            : data(data_), size(size_), allocator(allocator_)
        {
            CV_XADD(&data_->refcount, 1);
        }
        ~NumpyDeallocator()
        {
            if (allocator)
            {
                // This is called when the numpy array is deleted, thus we need to check if the cv::Mat's still hold a
                // reference to this memory
                // allocator->deallocateCpu(static_cast<uchar*>(data), size);
                if (data && CV_XADD(&data->refcount, -1) == 1)
                {
                    (data->currAllocator ? data->currAllocator : allocator ? allocator : cv::Mat::getDefaultAllocator())
                        ->unmap(data);
                }
            }
        }
        cv::UMatData* data;
        size_t size;
        const NumpyAllocator* allocator;
    };

    void setupAllocator()
    {
        boost::python::class_<NumpyDeallocator, boost::shared_ptr<NumpyDeallocator>, boost::noncopyable>(
            "NumpyDallocator", boost::python::no_init);
        import_array();
    }

    class PyEnsureGIL
    {
      public:
        PyEnsureGIL() : _state(PyGILState_Ensure()) {}
        ~PyEnsureGIL() { PyGILState_Release(_state); }
      private:
        PyGILState_STATE _state;
    };

    NumpyAllocator::NumpyAllocator(std::shared_ptr<Allocator> default_allocator_)
        : default_allocator(default_allocator_)
    {
    }

    NumpyAllocator::~NumpyAllocator() {}

    cv::Mat NumpyAllocator::fromPython(PyObject* arr) const {}

    PyObject* NumpyAllocator::toPython(const cv::Mat& mat) const
    {
        int type = mat.type();
        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t) / 8);
        int typenum = depth == CV_8U ? NPY_UBYTE
                                     : depth == CV_8S ? NPY_BYTE : depth == CV_16U
                                                                       ? NPY_USHORT
                                                                       : depth == CV_16S
                                                                             ? NPY_SHORT
                                                                             : depth == CV_32S
                                                                                   ? NPY_INT
                                                                                   : depth == CV_32F
                                                                                         ? NPY_FLOAT
                                                                                         : depth == CV_64F
                                                                                               ? NPY_DOUBLE
                                                                                               : f * NPY_ULONGLONG +
                                                                                                     (f ^ 1) * NPY_UINT;
        int dims = 0;
        if (mat.rows == 1 || mat.cols == 1)
            dims += 1;
        else
            dims += 2;
        if (cn != 1)
            dims += 1;

        size_t total_size = 0;
        cv::AutoBuffer<npy_intp, 10> _sizes(dims + 1);
        int i = 0;
        if (mat.rows > 1)
        {
            _sizes[i] = mat.rows;
            total_size += mat.rows;
            ++i;
        }
        if (mat.cols > 1)
        {
            _sizes[i] = mat.cols;
            total_size += mat.cols;
            ++i;
        }
        if (cn > 1)
        {
            _sizes[i] = cn;
            total_size += cn;
        }
        auto u = mat.u;
        PyEnsureGIL gil;
        PyObject* o = PyArray_SimpleNewFromData(dims, _sizes, typenum, mat.data);
        // PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        boost::python::object base(boost::shared_ptr<NumpyDeallocator>(new NumpyDeallocator(u, total_size, this)));
        PyArray_BASE(o) = base.ptr();
        Py_INCREF(base.ptr());
        u->userdata = o;
        return o;
    }

    cv::UMatData* NumpyAllocator::allocate(
        int dims0, const int* sizes, int type, void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
    {
        if (data != 0)
        {
            return static_cast<cv::MatAllocator*>(default_allocator.get())
                ->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        }
        auto ret = static_cast<cv::MatAllocator*>(default_allocator.get())
                       ->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        ret->currAllocator = this;
        return ret;
    }

    bool NumpyAllocator::allocate(cv::UMatData* u, int accessFlags, cv::UMatUsageFlags usageFlags) const
    {
        return static_cast<cv::MatAllocator*>(default_allocator.get())->allocate(u, accessFlags, usageFlags);
    }

    void NumpyAllocator::deallocate(cv::UMatData* u) const
    {
        if (u->userdata && u->currAllocator == this)
        {
            PyObject* o = static_cast<PyObject*>(u->userdata);
        }
        else
        {
        }
        default_allocator->deallocate(u);
    }

    // Used for stl allocators
    unsigned char* NumpyAllocator::allocateGpu(size_t num_bytes) { return default_allocator->allocateGpu(num_bytes); }
    void NumpyAllocator::deallocateGpu(uchar* ptr, size_t numBytes) { default_allocator->deallocateGpu(ptr, numBytes); }

    unsigned char* NumpyAllocator::allocateCpu(size_t num_bytes) { return default_allocator->allocateCpu(num_bytes); }
    void NumpyAllocator::deallocateCpu(uchar* ptr, size_t numBytes) { default_allocator->deallocateCpu(ptr, numBytes); }

    bool NumpyAllocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
    {
        return static_cast<cv::cuda::GpuMat::Allocator*>(default_allocator.get())->allocate(mat, rows, cols, elemSize);
    }

    void NumpyAllocator::free(cv::cuda::GpuMat* mat)
    {
        static_cast<cv::cuda::GpuMat::Allocator*>(default_allocator.get())->free(mat);
    }
}
