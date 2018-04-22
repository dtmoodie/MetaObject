#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
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

        template <>
        void convertFromPython(const boost::python::object& obj, cv::Mat& result)
        {
            PyObject* o = obj.ptr();
            if (PyArray_Check(o))
            {
                NumpyAllocator* alloc = dynamic_cast<NumpyAllocator*>(cv::Mat::getDefaultAllocator());
                if (alloc)
                {
                    result = alloc->fromPython(o);
                }
            }
        }
    }

    struct NumpyDeallocator
    {
        NumpyDeallocator(cv::UMatData* data_, const NumpyAllocator* allocator_) : data(data_), allocator(allocator_)
        {
            CV_XADD(&data_->refcount, 1);
        }

        ~NumpyDeallocator()
        {
            if (allocator)
            {
                if (data && CV_XADD(&data->refcount, -1) == 1)
                {
                    (data->currAllocator ? data->currAllocator : allocator ? allocator : cv::Mat::getDefaultAllocator())
                        ->unmap(data);
                }
            }
        }

        cv::UMatData* data;
        const NumpyAllocator* allocator;
    };

    int importNumpy()
    {
        import_array1(0);
        return 1;
    }

    void setupAllocator()
    {
        boost::python::class_<NumpyDeallocator, boost::shared_ptr<NumpyDeallocator>, boost::noncopyable>(
            "NumpyDallocator", boost::python::no_init);
        importNumpy();
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

    cv::Mat NumpyAllocator::fromPython(PyObject* o) const
    {
        cv::Mat m;

        bool allowND = true;
        if (!o || o == Py_None)
        {
            if (!m.data)
                m.allocator = const_cast<NumpyAllocator*>(this);
            return m;
        }

        if (PyLong_Check(o))
        {
            double v[] = {static_cast<double>(PyLong_AsLong((PyObject*)o)), 0., 0., 0.};
            m = cv::Mat(4, 1, CV_64F, v).clone();
            return m;
        }
        if (PyFloat_Check(o))
        {
            double v[] = {PyFloat_AsDouble((PyObject*)o), 0., 0., 0.};
            m = cv::Mat(4, 1, CV_64F, v).clone();
            return m;
        }
        if (PyTuple_Check(o))
        {
            int i, sz = (int)PyTuple_Size((PyObject *)o);
            m = cv::Mat(sz, 1, CV_64F);
            for (i = 0; i < sz; i++)
            {
                PyObject* oi = PyTuple_GET_ITEM(o, i);
                if (PyLong_Check(oi))
                    m.at<double>(i) = (double)PyLong_AsLong(oi);
                else if (PyFloat_Check(oi))
                    m.at<double>(i) = (double)PyFloat_AsDouble(oi);
                else
                {
                    m.release();
                    return m;
                }
            }
            return m;
        }

        if (!PyArray_Check(o))
        {
            return m;
        }

        PyArrayObject* oarr = (PyArrayObject*)o;

        bool needcopy = false, needcast = false;
        int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
        int type =
            typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE
                                               ? CV_8S
                                               : typenum == NPY_USHORT
                                                     ? CV_16U
                                                     : typenum == NPY_SHORT
                                                           ? CV_16S
                                                           : typenum == NPY_INT
                                                                 ? CV_32S
                                                                 : typenum == NPY_INT32
                                                                       ? CV_32S
                                                                       : typenum == NPY_FLOAT
                                                                             ? CV_32F
                                                                             : typenum == NPY_DOUBLE ? CV_64F : -1;

        if (type < 0)
        {
            if (typenum == NPY_INT64 || typenum == NPY_UINT64 || typenum == NPY_LONG)
            {
                needcopy = needcast = true;
                new_typenum = NPY_INT;
                type = CV_32S;
            }
            else
            {
                return m;
            }
        }

#ifndef CV_MAX_DIM
        const int CV_MAX_DIM = 32;
#endif

        int ndims = PyArray_NDIM(oarr);
        if (ndims >= CV_MAX_DIM)
        {
            return m;
        }

        int size[CV_MAX_DIM + 1];
        size_t step[CV_MAX_DIM + 1];
        size_t elemsize = CV_ELEM_SIZE1(type);
        const npy_intp* _sizes = PyArray_DIMS(oarr);
        const npy_intp* _strides = PyArray_STRIDES(oarr);
        bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

        for (int i = ndims - 1; i >= 0 && !needcopy; i--)
        {
            // these checks handle cases of
            //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
            //  b) transposed arrays, where _strides[] elements go in non-descending order
            //  c) flipped arrays, where some of _strides[] elements are negative
            // the _sizes[i] > 1 is needed to avoid spurious copies when NPY_RELAXED_STRIDES is set
            if ((i == ndims - 1 && _sizes[i] > 1 && (size_t)_strides[i] != elemsize) ||
                (i < ndims - 1 && _sizes[i] > 1 && _strides[i] < _strides[i + 1]))
                needcopy = true;
        }

        if (ismultichannel && _strides[1] != (npy_intp)elemsize * _sizes[2])
            needcopy = true;

        if (needcopy)
        {
            if (needcast)
            {
                o = PyArray_Cast(oarr, new_typenum);
                oarr = (PyArrayObject*)o;
            }
            else
            {
                oarr = PyArray_GETCONTIGUOUS(oarr);
                o = (PyObject*)oarr;
            }

            _strides = PyArray_STRIDES(oarr);
        }

        // Normalize strides in case NPY_RELAXED_STRIDES is set
        size_t default_step = elemsize;
        for (int i = ndims - 1; i >= 0; --i)
        {
            size[i] = (int)_sizes[i];
            if (size[i] > 1)
            {
                step[i] = (size_t)_strides[i];
                default_step = step[i] * size[i];
            }
            else
            {
                step[i] = default_step;
                default_step *= size[i];
            }
        }

        // handle degenerate case
        if (ndims == 0)
        {
            size[ndims] = 1;
            step[ndims] = elemsize;
            ndims++;
        }

        if (ismultichannel)
        {
            ndims--;
            type |= CV_MAKETYPE(0, size[2]);
        }

        if (ndims > 2 && !allowND)
        {
            return m;
        }

        m = cv::Mat(ndims, size, type, PyArray_DATA(oarr), step);
        cv::UMatData* u = new cv::UMatData(this);
        u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*)o);
        for (int i = 0; i < ndims - 1; i++)
            step[i] = (size_t)_strides[i];
        step[ndims - 1] = CV_ELEM_SIZE(type);
        u->size = size[0] * step[0];
        u->userdata = o;
        m.addref();

        if (!needcopy)
        {
            Py_INCREF(o);
        }
        m.allocator = const_cast<NumpyAllocator*>(this);
        return m;
    }

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
        boost::python::object base(boost::shared_ptr<NumpyDeallocator>(new NumpyDeallocator(u, this)));

        ((PyArrayObject_fields*)o)->base = base.ptr();
        // PyArray_BASE(o) = base.ptr();
        Py_INCREF(base.ptr());
        u->userdata = nullptr;
        return o;
    }

    cv::UMatData* NumpyAllocator::allocate(
        int dims0, const int* sizes, int type, void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
    {
        if (data != 0)
        {
            return dynamic_cast<cv::MatAllocator*>(default_allocator.get())
                ->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        }
        auto ret = dynamic_cast<cv::MatAllocator*>(default_allocator.get())
                       ->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        ret->currAllocator = this;
        return ret;
    }

    bool NumpyAllocator::allocate(cv::UMatData* u, int accessFlags, cv::UMatUsageFlags usageFlags) const
    {
        return dynamic_cast<cv::MatAllocator*>(default_allocator.get())->allocate(u, accessFlags, usageFlags);
    }

    void NumpyAllocator::deallocate(cv::UMatData* u) const
    {
        if (u->userdata && u->currAllocator == this)
        {
            PyObject* o = static_cast<PyObject*>(u->userdata);
            // May needa  GIL lock here, not sure.  If we crash, we crash...
            Py_DECREF(o);
        }
        else
        {
            dynamic_cast<cv::MatAllocator*>(default_allocator.get())->deallocate(u);
        }
    }
}
