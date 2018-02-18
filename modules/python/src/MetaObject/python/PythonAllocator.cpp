#include "PythonAllocator.hpp"
#include "MetaObject/logging/logging.hpp"

#include <numpy/ndarrayobject.h>

#include <boost/python.hpp>
namespace mo
{
    struct NumpyDeallocator
    {
        NumpyDeallocator(cv::UMatData* data_, size_t size_, const NumpyAllocator* allocator_)
            : data(data_), size(size_), allocator(allocator_)
        {
        }
        ~NumpyDeallocator()
        {
            if (allocator)
            {
                // allocator->deallocateCpu(static_cast<uchar*>(data), size);
            }
        }
        cv::UMatData* data;
        size_t size;
        const NumpyAllocator* allocator;
    };

    void setupAllocator()
    {
        boost::python::class_<NumpyDeallocator, boost::noncopyable>("NumpyDallocator", boost::python::no_init);
    }

    class PyEnsureGIL
    {
      public:
        PyEnsureGIL() : _state(PyGILState_Ensure()) {}
        ~PyEnsureGIL() { PyGILState_Release(_state); }
      private:
        PyGILState_STATE _state;
    };

    NumpyAllocator::NumpyAllocator(std::shared_ptr<Allocator> default_allocator_) {}

    NumpyAllocator::~NumpyAllocator() {}

    cv::UMatData* NumpyAllocator::allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const
    {
        cv::UMatData* u = new cv::UMatData(this);
        u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*)o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*)o);
        for (int i = 0; i < dims - 1; i++)
            step[i] = (size_t)_strides[i];
        step[dims - 1] = CV_ELEM_SIZE(type);
        u->size = sizes[0] * step[0];
        u->userdata = o;
        return u;
    }

    cv::UMatData* NumpyAllocator::allocate(
        int dims0, const int* sizes, int type, void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
    {
        if (data != 0)
        {
            // issue #6969: CV_Error(Error::StsAssert, "The data should normally be NULL!");
            // probably this is safe to do in such extreme case

            return static_cast<cv::MatAllocator*>(default_allocator.get())
                ->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        }
        PyEnsureGIL gil;

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
        int i, dims = dims0;
        size_t total_size = 0;
        cv::AutoBuffer<npy_intp> _sizes(dims + 1);
        for (i = 0; i < dims; i++)
        {
            _sizes[i] = sizes[i];
            total_size += sizes[i];
        }
        if (cn > 1)
            _sizes[dims++] = cn;

        cv::UMatData* ret = static_cast<cv::MatAllocator*>(default_allocator.get())
                                ->allocate(dims0, sizes, type, data, step, flags, usageFlags);

        MO_ASSERT(ret);
        PyObject* o = PyArray_SimpleNewFromData(dims, _sizes, typenum, ret->data);
        MO_ASSERT(o);
        boost::python::object base(NumpyDeallocator(ret, total_size, this));
        PyArray_BASE(o) = base.ptr();
        // cv::UMatData* u = new cv::UMatData(this);

        // ret->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*)o);
        // npy_intp* _strides = PyArray_STRIDES((PyArrayObject*)o);
        // for (int i = 0; i < dims - 1; i++)
        //    step[i] = (size_t)_strides[i];
        // step[dims - 1] = CV_ELEM_SIZE(type);
        // ret->size = sizes[0] * step[0];
        ret->userdata = o;
        ret->currAllocator = this;
        return ret;
        // PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        // if (!o)
        //    CV_Error_(Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        // return allocate(o, dims0, sizes, type, step);
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
