#pragma once
#ifndef __CUDACC__
#include "MetaObject/logging/logging.hpp"
#include <MetaObject/params/AccessToken.hpp>
#include <boost/thread/recursive_mutex.hpp>

namespace mo
{
    template <typename T>
    class TParamPtr;
    template <typename T, int N, typename Enable>
    struct MetaParam;

    template <typename T>
    TParamPtr<T>::TParamPtr(const std::string& name, T* ptr_, ParamFlags type, bool ownsData_)
        : ptr(ptr_), ownsData(ownsData_), ITParam<T>(name, type)
    {
        (void)&_meta_param;
    }

    template <typename T>
    TParamPtr<T>::~TParamPtr()
    {
        if (ownsData && ptr)
            delete ptr;
    }

    template <typename T>
    bool TParamPtr<T>::getData(InputStorage_t& value, const OptionalTime_t& ts, Context* ctx, size_t* fn_)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if (!ts)
        {
            if (ptr)
            {
                ParamTraits<T>::reset(value, *ptr);
                // value = *ptr;
                return true;
            }
        }
        else
        {
            if (this->_ts && *(this->_ts) == *ts && ptr)
            {
                ParamTraits<T>::reset(value, *ptr);
                return true;
            }
        }
        return false;
    }

    template <typename T>
    bool TParamPtr<T>::getData(InputStorage_t& value, size_t fn, Context* ctx, OptionalTime_t* ts)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if (fn == this->_fn && ptr)
        {
            ParamTraits<T>::reset(value, *ptr);
            if (ts)
            {
                *ts = this->_ts;
            }
            return true;
        }
        return false;
    }

    template <typename T>
    IParam* TParamPtr<T>::emitUpdate(const OptionalTime_t& ts_,
                                     Context* ctx_,
                                     const boost::optional<size_t>& fn_,
                                     const std::shared_ptr<ICoordinateSystem>& cs_,
                                     UpdateFlags flags_)
    {
        IParam::emitUpdate(ts_, ctx_, fn_, cs_, flags_);
        if (ptr)
        {
            ITParam<T>::_typed_update_signal(ParamTraits<T>::copy(*ptr), this, ctx_, ts_, this->_fn, cs_, flags_);
        }
        return this;
    }

    template <typename T>
    AccessToken<T> TParamPtr<T>::access()
    {
        MO_ASSERT(ptr);
        return AccessToken<T>(*this, *ptr);
    }

    template <typename T>
    ConstAccessToken<T> TParamPtr<T>::access() const
    {
        MO_ASSERT(ptr);
        return ConstAccessToken<T>(*this, ParamTraits<T>::get(*ptr));
    }

    template <typename T>
    bool TParamPtr<T>::updateDataImpl(const Storage_t& data,
                                      const OptionalTime_t& ts,
                                      Context* ctx,
                                      size_t fn,
                                      const std::shared_ptr<ICoordinateSystem>& cs)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if (ptr)
        {
            *ptr = ParamTraits<T>::get(data);
            lock.unlock();
            this->emitUpdate(ts, ctx, fn, cs);
            ITParam<T>::_typed_update_signal(data, this, ctx, ts, fn, cs, mo::ValueUpdated_e);
            return true;
        }
        return false;
    }

    template <typename T>
    ITParam<T>* TParamPtr<T>::updatePtr(Raw_t* ptr, bool ownsData_)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        this->ptr = ptr;
        this->ownsData = ownsData_;
        return this;
    }

    template <typename T>
    MetaParam<T, 100, void> TParamPtr<T>::_meta_param;

    template <typename T>
    class MO_EXPORTS TParamPtr<std::shared_ptr<T>> : virtual public ITAccessibleParam<T>
    {
      public:
        typedef typename ParamTraits<T>::Storage_t Storage_t;
        typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
        typedef typename ParamTraits<T>::InputStorage_t InputStorage_t;
        typedef typename ParamTraits<T>::Input_t Input_t;
        typedef typename ParamTraits<T>::Raw_t Raw_t;
        typedef void(TUpdateSig_t)(ConstStorageRef_t,
                                   IParam*,
                                   Context*,
                                   OptionalTime_t,
                                   size_t,
                                   const std::shared_ptr<ICoordinateSystem>&,
                                   UpdateFlags);
        typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
        typedef TSlot<TUpdateSig_t> TUpdateSlot_t;

        /*!
         * \brief TParamPtr default constructor
         * \param name of the Param
         * \param ptr_ to user owned data
         * \param type of Param
         * \param ownsData_ cleanup on delete?
         */
        TParamPtr(const std::string& name = "",
                  std::shared_ptr<T>* ptr_ = nullptr,
                  ParamFlags type = ParamFlags::Control_e,
                  bool ownsData_ = false)
            : ptr(ptr_), ownsData(ownsData_), ITParam<T>(name, type)
        {
            (void)_meta_param;
        }
        ~TParamPtr()
        {
            if (ownsData && ptr)
                delete ptr;
        }

        virtual bool getData(InputStorage_t& data,
                             const OptionalTime_t& ts = OptionalTime_t(),
                             Context* ctx = nullptr,
                             size_t* fn_ = nullptr)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (!ts)
            {
                if (ptr)
                {
                    data = *ptr;
                    return true;
                }
            }
            else
            {
                if (this->_ts && *(this->_ts) == *ts && ptr)
                {
                    data = *ptr;
                    return true;
                }
            }
            return false;
        }

        virtual bool getData(InputStorage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (fn == this->_fn && ptr)
            {
                // ParamTraits<T>::reset(data, *ptr);
                data = *ptr;
                if (ts_)
                {
                    *ts_ = this->_ts;
                }
                return true;
            }
            return false;
        }

        virtual IParam* emitUpdate(const OptionalTime_t& ts_ = OptionalTime_t(),
                                   Context* ctx_ = mo::Context::getCurrent(),
                                   const boost::optional<size_t>& fn_ = boost::optional<size_t>(),
                                   const std::shared_ptr<ICoordinateSystem>& cs_ = std::shared_ptr<ICoordinateSystem>(),
                                   UpdateFlags flags_ = ValueUpdated_e)
        {
            IParam::emitUpdate(ts_, ctx_, fn_, cs_, flags_);
            if (ptr)
            {
                ITParam<T>::_typed_update_signal(ParamTraits<T>::copy(*ptr), this, ctx_, ts_, this->_fn, cs_, flags_);
            }
            return this;
        }

        virtual IParam* emitUpdate(const IParam& other) { return IParam::emitUpdate(other); }

        virtual AccessToken<T> access()
        {
            MO_ASSERT(ptr);
            return AccessToken<T>(*this, ParamTraits<T>::get(*ptr));
        }

        virtual ConstAccessToken<T> access() const
        {
            MO_ASSERT(ptr);
            return ConstAccessToken<T>(*this, ParamTraits<T>::get(*ptr));
        }

        ITParam<T>* updatePtr(std::shared_ptr<T>* ptr, bool ownsData_ = false)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            this->ptr = ptr;
            this->ownsData = ownsData_;
            return this;
        }

      protected:
        virtual bool updateDataImpl(const Storage_t& data,
                                    const OptionalTime_t& ts,
                                    Context* ctx,
                                    size_t fn,
                                    const std::shared_ptr<ICoordinateSystem>& cs)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (ptr)
            {
                *ptr = data;
                lock.unlock();
                this->emitUpdate(ts, ctx, fn, cs);
                ITParam<T>::_typed_update_signal(data, this, ctx, ts, fn, cs, mo::ValueUpdated_e);
                return true;
            }
            return false;
        }
        std::shared_ptr<T>* ptr;
        bool ownsData;
        static MetaParam<T, 100> _meta_param;
    };

    template <typename T>
    MetaParam<T, 100, void> TParamPtr<std::shared_ptr<T>>::_meta_param;

    template <typename T>
    bool TParamOutput<T>::getData(InputStorage_t& data, const OptionalTime_t& ts, Context* ctx, size_t* fn_)
    {
        if (!ts || ts == this->_ts)
        {
            data = this->data;
            if (fn_)
            {
                *fn_ = this->_fn;
            }
            return true;
        }
        return false;
    }

    template <typename T>
    bool TParamOutput<T>::getData(InputStorage_t& data, size_t fn, Context* ctx, OptionalTime_t* ts_)
    {
        if (fn == this->_fn)
        {
            data = this->data;
            if (ts_)
            {
                *ts_ = this->_ts;
            }
            return true;
        }
        return false;
    }

    template <typename T>
    AccessToken<T> TParamOutput<T>::access()
    {
        return AccessToken<T>(*this, data);
    }

    template <typename T>
    ConstAccessToken<T> TParamOutput<T>::access() const
    {
        return ConstAccessToken<T>(*this, ParamTraits<T>::get(data));
    }

    template <typename T>
    bool TParamOutput<T>::updateDataImpl(const Storage_t& data,
                                         const OptionalTime_t& ts,
                                         Context* ctx,
                                         size_t fn,
                                         const std::shared_ptr<ICoordinateSystem>& cs)
    {
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        // this->data = data;
        if (this->ptr)
        {
            *(this->ptr) = ParamTraits<T>::get(data);
        }
        lock.unlock();
        this->emitUpdate(ts, ctx, fn, cs);
        return true;
    }

    template <typename T>
    IParam* TParamOutput<T>::emitUpdate(const OptionalTime_t& ts_,
                                        Context* ctx_,
                                        const boost::optional<size_t>& fn_,
                                        const std::shared_ptr<ICoordinateSystem>& cs_,
                                        UpdateFlags flags_)
    {
        if (this->ptr)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (this->checkFlags(mo::ParamFlags::Unstamped_e))
            {
                data = ParamTraits<T>::copy(*(this->ptr));
            }
            else
            {
                ParamTraits<T>::move(data, std::move(*(this->ptr)));
            }
            lock.unlock();
            IParam::emitUpdate(ts_, ctx_, fn_, cs_, flags_);
            ITParam<T>::_typed_update_signal(data, this, ctx_, ts_, this->_fn, cs_, flags_);
        }

        return this;
    }

    template <typename T>
    class MO_EXPORTS TParamOutput<std::shared_ptr<T>> : virtual public TParamPtr<std::shared_ptr<T>>
    {
      public:
        typedef typename ParamTraits<T>::Storage_t Storage_t;
        typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
        typedef typename ParamTraits<T>::InputStorage_t InputStorage_t;
        typedef typename ParamTraits<T>::Input_t Input_t;
        typedef void(TUpdateSig_t)(ConstStorageRef_t,
                                   IParam*,
                                   Context*,
                                   OptionalTime_t,
                                   size_t,
                                   const std::shared_ptr<ICoordinateSystem>&,
                                   UpdateFlags);
        typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
        typedef TSlot<TUpdateSig_t> TUpdateSlot_t;

        TParamOutput() : IParam(mo::tag::_param_flags = mo::ParamFlags::Output_e) {}

        virtual bool getData(InputStorage_t& data,
                             const OptionalTime_t& ts = OptionalTime_t(),
                             Context* ctx = nullptr,
                             size_t* fn_ = nullptr)
        {
            if (!ts || ts == this->_ts)
            {
                data = this->data;
                if (fn_)
                {
                    *fn_ = this->_fn;
                }
                return true;
            }
            return false;
        }

        virtual bool getData(InputStorage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr)
        {
            if (fn == this->_fn)
            {
                data = this->data;
                if (ts_)
                {
                    *ts_ = this->_ts;
                }
                return true;
            }
            return false;
        }

        virtual AccessToken<T> access() { return AccessToken<T>(*this, data); }

        virtual ConstAccessToken<T> access() const { return ConstAccessToken<T>(*this, ParamTraits<T>::get(data)); }

        virtual IParam* emitUpdate(const OptionalTime_t& ts_ = OptionalTime_t(),
                                   Context* ctx_ = mo::Context::getCurrent(),
                                   const boost::optional<size_t>& fn_ = boost::optional<size_t>(),
                                   const std::shared_ptr<ICoordinateSystem>& cs_ = std::shared_ptr<ICoordinateSystem>(),
                                   UpdateFlags flags_ = ValueUpdated_e)
        {
            if (this->ptr)
            {
                mo::Mutex_t::scoped_lock lock(IParam::mtx());
                data = *this->ptr;
                lock.unlock();
                IParam::emitUpdate(ts_, ctx_, fn_, cs_, flags_);
                ITParam<T>::_typed_update_signal(data, this, ctx_, ts_, this->_fn, cs_, flags_);
            }

            return this;
        }
        virtual IParam* emitUpdate(const IParam& other) { return IParam::emitUpdate(other); }

      protected:
        virtual bool updateDataImpl(const Storage_t& data,
                                    const OptionalTime_t& ts,
                                    Context* ctx,
                                    size_t fn,
                                    const std::shared_ptr<ICoordinateSystem>& cs)
        {
            mo::Mutex_t::scoped_lock lock(IParam::mtx());
            if (this->ptr)
            {
                *(this->ptr) = data;
            }
            lock.unlock();
            this->emitUpdate(ts, ctx, fn, cs);
            return true;
        }

      private:
        Storage_t data;
    };
}
#endif
