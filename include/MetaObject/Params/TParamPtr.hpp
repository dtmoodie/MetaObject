#pragma once
#include "ITParam.hpp"
#include "MetaParam.hpp"
namespace mo {
/*! The TParamPtr class is a concrete implementation of ITParam
 *  which implements wrapping of a raw pointer to user data.  This is used
 *  extensively inside of the OUTPUT macro as follows:
 *
 *  float user_data;
 *  TParamPtr<float> user_param("float_data", &user_data);
 *  user_param.UpdateData(10);
 *
 *  This code snipit creates a user space variable 'user_data'
 *  which is wrapped for reflection purposes by 'user_param'
 */
template<typename T>
class MO_EXPORTS TParamPtr: virtual public ITParam< T > {
public:
    /*!
     * \brief TParamPtr default constructor
     * \param name of the Param
     * \param ptr_ to user owned data
     * \param type of Param
     * \param ownsData_ cleanup on delete?
     */
    TParamPtr(const std::string& name = "",
                      T* ptr_ = nullptr,
                      ParamFlags type = Control_e,
                      bool ownsData_ = false);
    ~TParamPtr();

    virtual bool getData(Storage_t& data, const OptionalTime_t& ts = OptionalTime_t(),
        Context* ctx = nullptr, size_t* fn_ = nullptr);

    virtual bool getData(Storage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);

    ITParam<T>* updatePtr(T* ptr, bool ownsData_ = false);
protected:
    virtual bool updateDataImpl(ConstStorageRef_t data, OptionalTime_t ts, Context* ctx, size_t fn, ICoordinateSystem* cs);
    //virtual bool updateDataImpl(const T& data, OptionalTime_t ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs);
    T* ptr;
    bool ownsData;
    static MetaParam<T, 100> _meta_Param;
};
}
//#include "detail/TParamPtrImpl.hpp"
