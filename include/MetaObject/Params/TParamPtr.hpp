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

    T*   GetDataPtr(OptionalTime_t ts = OptionalTime_t(), Context* ctx = nullptr, size_t* fn_ = nullptr);
    T*   GetDataPtr(size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);
    T    GetData(OptionalTime_t ts = OptionalTime_t(), Context* ctx = nullptr, size_t* fn = nullptr);
    T    GetData(size_t fn, Context* ctx = nullptr, OptionalTime_t* ts = nullptr);
    bool GetData(T& value, OptionalTime_t ts = OptionalTime_t(), Context* ctx = nullptr, size_t* fn = nullptr);
    bool GetData(T& value, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts = nullptr);

    /*virtual ITParam<T>* UpdateData(const T& data,
                                           mo::Time_t ts = -1 * mo::second,
                                           Context* ctx = nullptr,
                                           size_t fn = std::numeric_limits<size_t>::max(),
                                           ICoordinateSystem* cs = nullptr);*/

    virtual bool Update(IParam* other);
    virtual std::shared_ptr<IParam> DeepCopy() const;
    ITParam<T>* UpdatePtr(T* ptr, bool ownsData_ = false);
protected:
    virtual bool UpdateDataImpl(const T& data, OptionalTime_t ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs);
    T* ptr;
    bool ownsData;
    static MetaParam<T, 100> _meta_Param;
};
}
//#include "detail/TParamPtrImpl.hpp"
