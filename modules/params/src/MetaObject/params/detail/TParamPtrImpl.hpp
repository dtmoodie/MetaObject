#pragma once
#ifndef __CUDACC__
#include "../TParamPtr.hpp"
#include "MetaObject/logging/logging.hpp"
#include "print_data.hpp"
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
        : m_ptr(ptr_)
        , ownsData(ownsData_)
        , ITParam<T>(name, type)
    {
    }

    template <typename T>
    TParamPtr<T>::~TParamPtr()
    {
        if (ownsData && m_ptr)
        {
            delete m_ptr;
        }
        else
        {
        }
    }

    template <typename T>
    bool TParamPtr<T>::getData(InputStorage_t& value, const OptionalTime_t& ts, Context* /*ctx*/, size_t* fn_)
    {
        Lock lock(IParam::mtx());
        if (!ts)
        {
            if (m_ptr)
            {
                ParamTraits<T>::reset(value, *m_ptr);
                if (fn_)
                {
                    *fn_ = IParam::getFrameNumber();
                }
                else
                {
                }
                return true;
            }
            else
            {
            }
        }
        else
        {
            if (this->_ts && (*(this->_ts) == *ts) && m_ptr)
            {
                ParamTraits<T>::reset(value, *m_ptr);
                if (fn_)
                {
                    *fn_ = IParam::getFrameNumber();
                }
                else
                {
                }
                return true;
            }
            else
            {
            }
        }
        return false;
    }

    template <typename T>
    bool TParamPtr<T>::getData(InputStorage_t& value, size_t fn, Context* /*ctx*/, OptionalTime_t* ts)
    {
        Lock lock(IParam::mtx());
        if ((fn == this->_fn) && m_ptr)
        {
            ParamTraits<T>::reset(value, *m_ptr);
            if (ts)
            {
                *ts = this->_ts;
            }
            else
            {
            }
            return true;
        }
        else
        {
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
        if (m_ptr)
        {
            ITParamImpl<T>::emitTypedUpdate(ParamTraits<T>::copy(*m_ptr), this, ctx_, ts_, this->_fn, cs_, flags_);
        }
        return this;
    }

    template <typename T>
    AccessToken<T> TParamPtr<T>::access()
    {
        MO_ASSERT(m_ptr);
        return AccessToken<T>(*this, *m_ptr);
    }

    template <typename T>
    ConstAccessToken<T> TParamPtr<T>::read() const
    {
        MO_ASSERT(m_ptr);
        return ConstAccessToken<T>(*this, ParamTraits<T>::get(*m_ptr));
    }

    template <typename T>
    std::ostream& TParamPtr<T>::print(std::ostream& os) const
    {
        Lock lock(this->mtx());
        mo::IParam::print(os);
        os << ' ';
        if (m_ptr)
        {
            mo::print(os, *m_ptr);
        }
        return os;
    }

    template <typename T>
    bool TParamPtr<T>::updateDataImpl(const Storage_t& data,
                                      const OptionalTime_t& ts,
                                      Context* ctx,
                                      size_t fn,
                                      const std::shared_ptr<ICoordinateSystem>& cs)
    {
        Lock lock(IParam::mtx());
        if (m_ptr)
        {
            *m_ptr = ParamTraits<T>::get(data);
            lock.unlock();
            this->emitUpdate(ts, ctx, fn, cs);
            ITParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::ValueUpdated_e);
            return true;
        }
        else
        {
        }
        return false;
    }

    template <typename T>
    bool TParamPtr<T>::updateDataImpl(Storage_t&& data,
                                      const OptionalTime_t& ts,
                                      Context* ctx,
                                      size_t fn,
                                      const std::shared_ptr<ICoordinateSystem>& cs)
    {
        Lock lock(IParam::mtx());
        if (m_ptr)
        {
            *m_ptr = ParamTraits<T>::get(data);
            lock.unlock();
            this->emitUpdate(ts, ctx, fn, cs);
            ITParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::ValueUpdated_e);
            return true;
        }
        else
        {
        }
        return false;
    }

    template <typename T>
    ITParam<T>* TParamPtr<T>::updatePtr(Raw_t* ptr, bool ownsData_)
    {
        Lock lock(IParam::mtx());
        this->m_ptr = ptr;
        this->ownsData = ownsData_;
        return this;
    }

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
            : m_ptr(ptr_)
            , ownsData(ownsData_)
            , ITParam<T>(name, type)
        {
        }
        ~TParamPtr()
        {
            if (ownsData && m_ptr)
            {
                delete m_ptr;
            }
        }

        virtual bool getData(InputStorage_t& data,
                             const OptionalTime_t& ts = OptionalTime_t(),
                             Context* /*ctx*/ = nullptr,
                             size_t* fn_ = nullptr)
        {
            Lock lock(IParam::mtx());
            if (!ts)
            {
                if (m_ptr)
                {
                    data = *m_ptr;
                    if (fn_ != nullptr)
                    {
                        *fn_ = this->getFrameNumber();
                    }
                    else
                    {
                    }
                    return true;
                }
            }
            else
            {
                if (this->_ts && *(this->_ts) == *ts && m_ptr)
                {
                    data = *m_ptr;
                    return true;
                }
            }
            return false;
        }

        virtual bool getData(InputStorage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr)
        {
            Lock lock(IParam::mtx());
            if (fn == this->_fn && m_ptr)
            {
                // ParamTraits<T>::reset(data, *ptr);
                data = *m_ptr;
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
            if (m_ptr)
            {
                ITParam<T>::emitTypedUpdate(ParamTraits<T>::copy(*m_ptr), this, ctx_, ts_, this->_fn, cs_, flags_);
            }
            return this;
        }

        virtual IParam* emitUpdate(const IParam& other)
        {
            return IParam::emitUpdate(other);
        }

        virtual AccessToken<T> access()
        {
            MO_ASSERT(m_ptr);
            return AccessToken<T>(*this, ParamTraits<T>::get(*m_ptr));
        }

        virtual ConstAccessToken<T> read() const
        {
            MO_ASSERT(m_ptr);
            return ConstAccessToken<T>(*this, ParamTraits<T>::get(*m_ptr));
        }

        ITParam<T>* updatePtr(std::shared_ptr<T>* ptr, bool ownsData_ = false)
        {
            Lock lock(IParam::mtx());
            this->m_ptr = ptr;
            this->ownsData = ownsData_;
            return this;
        }
        std::shared_ptr<T>* ptr()
        {
            return m_ptr;
        }

      protected:
        bool updateDataImpl(const Storage_t& data,
                            const OptionalTime_t& ts,
                            Context* ctx,
                            size_t fn,
                            const std::shared_ptr<ICoordinateSystem>& cs) override
        {
            Lock lock(IParam::mtx());
            if (m_ptr)
            {
                *m_ptr = data;
                lock.unlock();
                this->emitUpdate(ts, ctx, fn, cs);
                ITParamImpl<T>::emitTypedUpdate(data, this, ctx, ts, fn, cs, mo::ValueUpdated_e);
                return true;
            }
            return false;
        }

        bool updateDataImpl(Storage_t&& data,
                            const OptionalTime_t& ts,
                            Context* ctx,
                            size_t fn,
                            const std::shared_ptr<ICoordinateSystem>& cs) override
        {
            Lock lock(IParam::mtx());
            if (m_ptr)
            {
                *m_ptr = std::move(data);
                lock.unlock();
                this->emitUpdate(ts, ctx, fn, cs);
                ITParamImpl<T>::emitTypedUpdate(*m_ptr, this, ctx, ts, fn, cs, mo::ValueUpdated_e);
                return true;
            }
            return false;
        }

      private:
        std::shared_ptr<T>* m_ptr;
        bool ownsData;
    };

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
            else
            {
            }
            return true;
        }
        else
        {
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
            else
            {
            }
            return true;
        }
        else
        {
        }
        return false;
    }

    template <typename T>
    AccessToken<T> TParamOutput<T>::access()
    {
        return AccessToken<T>(*this, data);
    }

    template <typename T>
    ConstAccessToken<T> TParamOutput<T>::read() const
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
        Lock lock(IParam::mtx());
        // this->data = data;
        auto ptr = this->ptr();
        if (ptr)
        {
            *(ptr) = ParamTraits<T>::get(data);
        }
        else
        {
        }
        lock.unlock();
        this->emitUpdate(ts, ctx, fn, cs);
        return true;
    }

    template <typename T>
    bool TParamOutput<T>::updateDataImpl(Storage_t&& data,
                                         const OptionalTime_t& ts,
                                         Context* ctx,
                                         size_t fn,
                                         const std::shared_ptr<ICoordinateSystem>& cs)
    {
        Lock lock(IParam::mtx());

        auto ptr = this->ptr();
        if (ptr)
        {
            ParamTraits<T>::move(*ptr, std::move(data));
        }
        else
        {
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
        auto ptr = this->ptr();
        if (ptr)
        {
            Lock lock(IParam::mtx());
            if (this->checkFlags(mo::ParamFlags::Unstamped_e))
            {
                data = ParamTraits<T>::copy(*(ptr));
            }
            else
            {
                ParamTraits<T>::move(data, std::move(*(ptr)));
            }
            lock.unlock();
            IParam::emitUpdate(ts_, ctx_, fn_, cs_, flags_);
            ITParamImpl<T>::emitTypedUpdate(data, this, ctx_, ts_, this->_fn, cs_, flags_);
        }

        return this;
    }

    template <typename T>
    std::vector<TypeInfo> TParamOutput<T>::listOutputTypes() const
    {
        return {getTypeInfo()};
    }

    template <typename T>
    ParamBase* TParamOutput<T>::getOutputParam()
    {
        return this;
    }

    template <typename T>
    ParamBase* TParamOutput<T>::getOutputParam(const TypeInfo type)
    {
        if (type == getTypeInfo())
        {
            return this;
        }
        else
        {
            return nullptr;
        }
    }

    template <typename T>
    const ParamBase* TParamOutput<T>::getOutputParam() const
    {
        return this;
    }

    template <typename T>
    const ParamBase* TParamOutput<T>::getOutputParam(const TypeInfo type) const
    {
        if (type == getTypeInfo())
        {
            return this;
        }
        else
        {
            return nullptr;
        }
    }

    template <typename T>
    class MO_EXPORTS TParamOutput<std::shared_ptr<T>> : virtual public TParamPtr<std::shared_ptr<T>>,
                                                        virtual public OutputParam
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

        TParamOutput()
            : IParam(mo::tag::_param_flags = mo::ParamFlags::Output_e)
        {
        }

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
                else
                {
                }
                return true;
            }
            else
            {
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
                else
                {
                }
                return true;
            }
            else
            {
            }
            return false;
        }

        virtual AccessToken<T> access()
        {
            return AccessToken<T>(*this, data);
        }

        virtual ConstAccessToken<T> read() const override
        {
            return ConstAccessToken<T>(*this, ParamTraits<T>::get(data));
        }

        bool canAccess() const override
        {
            return data != nullptr;
        }

        virtual IParam* emitUpdate(const OptionalTime_t& ts_ = OptionalTime_t(),
                                   Context* ctx_ = mo::Context::getCurrent(),
                                   const boost::optional<size_t>& fn_ = boost::optional<size_t>(),
                                   const std::shared_ptr<ICoordinateSystem>& cs_ = std::shared_ptr<ICoordinateSystem>(),
                                   UpdateFlags flags_ = ValueUpdated_e)
        {
            auto ptr = this->ptr();
            if (ptr)
            {
                Lock lock(IParam::mtx());
                data = *ptr;
                lock.unlock();
                IParam::emitUpdate(ts_, ctx_, fn_, cs_, flags_);
                ITParam<T>::emitTypedUpdate(data, this, ctx_, ts_, this->_fn, cs_, flags_);
            }
            else
            {
            }

            return this;
        }
        virtual IParam* emitUpdate(const IParam& other)
        {
            return IParam::emitUpdate(other);
        }

        std::vector<TypeInfo> listOutputTypes() const override
        {
            return {this->getTypeInfo()};
        }

        ParamBase* getOutputParam(const TypeInfo type) override
        {
            if (type == this->getTypeInfo())
            {
                return this;
            }
            else
            {
                return nullptr;
            }
        }

        ParamBase* getOutputParam() override
        {
            return this;
        }

        const ParamBase* getOutputParam(const TypeInfo type) const override
        {
            if (type == this->getTypeInfo())
            {
                return this;
            }
            else
            {
                return nullptr;
            }
        }

        const ParamBase* getOutputParam() const override
        {
            return this;
        }

      protected:
        bool updateDataImpl(const Storage_t& data,
                            const OptionalTime_t& ts,
                            Context* ctx,
                            size_t fn,
                            const std::shared_ptr<ICoordinateSystem>& cs) override
        {
            Lock lock(IParam::mtx());
            auto ptr = this->ptr();
            if (ptr)
            {
                *(ptr) = data;
            }
            else
            {
            }
            lock.unlock();
            this->emitUpdate(ts, ctx, fn, cs);
            return true;
        }

        bool updateDataImpl(Storage_t&& data,
                            const OptionalTime_t& ts,
                            Context* ctx,
                            size_t fn,
                            const std::shared_ptr<ICoordinateSystem>& cs) override
        {
            Lock lock(IParam::mtx());
            auto ptr = this->ptr();
            if (ptr)
            {
                *(ptr) = std::move(data);
            }
            else
            {
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
