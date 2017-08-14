#pragma once
#include "ITAccessibleParam.hpp"
#include "MetaParam.hpp"
namespace mo {
/*! The TParamPtr class is a concrete implementation of ITParam
 *  which implements wrapping of a raw pointer to user data.  This is used
 *  extensively inside of the PARAM macro as follows:
 *
 *  float user_data;
 *  TParamPtr<float> user_param("float_data", &user_data);
 *  user_param.updateData(10);
 *  user_data == 10;
 *
 *  This code snipit creates a user space variable 'user_data'
 *  which is wrapped for reflection purposes by 'user_param' named 'float_data'.
 *  Updates to user_param are reflected in user_data
 */
template <typename T>
class MO_EXPORTS TParamPtr : virtual public ITAccessibleParam<T> {
public:
    typedef typename ParamTraits<T>::Storage_t         Storage_t;
    typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
    typedef typename ParamTraits<T>::InputStorage_t    InputStorage_t;
    typedef typename ParamTraits<T>::Input_t           Input_t;
    typedef typename ParamTraits<T>::Raw_t             Raw_t;
    typedef void(TUpdateSig_t)(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags);
    typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
    typedef TSlot<TUpdateSig_t>   TUpdateSlot_t;

    /*!
     * \brief TParamPtr default constructor
     * \param name of the Param
     * \param ptr_ to user owned data
     * \param type of Param
     * \param ownsData_ cleanup on delete?
     */
    TParamPtr(const std::string& name      = "",
        T*                       ptr_      = nullptr,
        ParamFlags               type      = ParamFlags::Control_e,
        bool                     ownsData_ = false);
    ~TParamPtr();

    virtual bool getData(InputStorage_t& data, const OptionalTime_t& ts = OptionalTime_t(),
        Context* ctx = nullptr, size_t* fn_ = nullptr);

    virtual bool getData(InputStorage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

    virtual IParam* emitUpdate(const OptionalTime_t& ts_    = OptionalTime_t(),
        Context*                                     ctx_   = Context::getDefaultThreadContext().get(),
        const boost::optional<size_t>&               fn_    = boost::optional<size_t>(),
        const std::shared_ptr<ICoordinateSystem>&    cs_    = std::shared_ptr<ICoordinateSystem>(),
        UpdateFlags                                  flags_ = ValueUpdated_e);
    virtual IParam* emitUpdate(const IParam& other) { return IParam::emitUpdate(other); }
    virtual AccessToken<T>                   access();

    ITParam<T>* updatePtr(Raw_t* ptr, bool ownsData_ = false);

protected:
    virtual bool updateDataImpl(const Storage_t& data, const OptionalTime_t& ts, Context* ctx, size_t fn, const std::shared_ptr<ICoordinateSystem>& cs);
    Raw_t* ptr;
    bool   ownsData;
    static MetaParam<T, 100> _meta_Param;
};

/*!
 * TParamOutput is used with the OUTPUT macro.  In this case, the param owns the data and the owning parent object
 * owns a reference to the data which is updated by the param's reset function.
 */
template <typename T>
class MO_EXPORTS TParamOutput : virtual public TParamPtr<T> {
public:
    typedef typename ParamTraits<T>::Storage_t         Storage_t;
    typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
    typedef typename ParamTraits<T>::InputStorage_t    InputStorage_t;
    typedef typename ParamTraits<T>::Input_t           Input_t;
    typedef void(TUpdateSig_t)(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags);
    typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
    typedef TSlot<TUpdateSig_t>   TUpdateSlot_t;

    TParamOutput()
        : IParam(mo::tag::_param_flags = mo::ParamFlags::Output_e) {
    }

    virtual bool getData(InputStorage_t& data, const OptionalTime_t& ts = OptionalTime_t(),
        Context* ctx = nullptr, size_t* fn_ = nullptr);

    virtual bool getData(InputStorage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

    virtual AccessToken<T> access();

    virtual IParam* emitUpdate(const OptionalTime_t& ts_    = OptionalTime_t(),
        Context*                                     ctx_   = Context::getDefaultThreadContext().get(),
        const boost::optional<size_t>&               fn_    = boost::optional<size_t>(),
        const std::shared_ptr<ICoordinateSystem>&    cs_    = std::shared_ptr<ICoordinateSystem>(),
        UpdateFlags                                  flags_ = ValueUpdated_e);
    virtual IParam* emitUpdate(const IParam& other) { return IParam::emitUpdate(other); }

protected:
    virtual bool updateDataImpl(const Storage_t& data, const OptionalTime_t& ts, Context* ctx, size_t fn, const std::shared_ptr<ICoordinateSystem>& cs);

private:
    Storage_t data;
};
}
#include "detail/TParamPtrImpl.hpp"
