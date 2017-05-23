#pragma once
#include "MetaObject/logging/Log.hpp"
#include <type_traits>
namespace mo{
template<class Type, class Enable> struct ParamTraitsImpl;
/*!
 *  Specialization for non POD datatypes.  Assume it is necessary to store it in a std::shared_ptr
 *  share one object using std::shared_ptr<const Type> between the output, inputs, and buffers
 */
template<class Type> struct ParamTraitsImpl<Type, typename std::enable_if<!std::is_pod<Type>::value>::type> {
    enum {
        REQUIRES_GPU_SYNC = 0,
        HAS_TRIVIAL_MOVE = 0
    };
    // Raw original datatype. Used in TParamPtr when userspace variable is wrapped by a param
    typedef Type Raw_t;
    typedef std::shared_ptr<Type> Storage_t; // Used in output wrapping parameters
    // Used by output parameters where the member is really just a reference to what
    // is owned as a Storage_t by the parameter
    typedef Type& TypeRef_t;
    typedef const Type& ConstTypeRef_t;

    // Pointer to typed stored by storage
    typedef Type* StoragePtr_t;
    typedef const Type* ConstStoragePtr_t;

    // Used when passing data around within a thread
    typedef const std::shared_ptr<const Type>& ConstStorageRef_t;

    // Used for input parameters
    // Wrapping param storage
    typedef std::shared_ptr<const Type> InputStorage_t;
    // User space input pointer, used in TInputParamPtr
    typedef const Type* Input_t;


    // Shallow copy if possible
    static inline Storage_t copy(const Type& value) {
        return Storage_t(new Type(value));
    }
    static inline Storage_t copy(ConstStorageRef_t value){
        return value; // copy constructor in Storage_t
    }

    // deep copy if possible
    static Storage_t clone(const Type& value){
        Storage_t output(new Type(value)); // This assumes a copy constructor
        return output;
    }
    static Storage_t clone(ConstStorageRef_t  value){
        Storage_t output(new Type(get(value))); // This assumes a copy constructor
        return output;
    }

    template<class...Args>
    static Type& reset(Storage_t& input_storage, Args&&...args) {
        input_storage.reset(new Type(std::forward<Args>(args)...));
        return *input_storage;
    }

    template<class...Args>
    static void reset(InputStorage_t& input_storage, Args&&...args) {
        input_storage.reset(new Type(std::forward<Args>(args)...));
    }

    template<class...Args>
    static void nullify(InputStorage_t& input_storage) {
        input_storage.reset();
    }

    // Access underlying data
    static inline Type& get(const Storage_t& value){
        MO_ASSERT(value);
        return *value;
    }

    static inline const Type& get(ConstStorageRef_t& value) {
        MO_ASSERT(value);
        return *value;
    }

    static inline StoragePtr_t ptr(const Storage_t& data){
        return data.get();
    }
    static inline ConstStoragePtr_t ptr(ConstStorageRef_t& data) {
        return data.get();
    }

};

} // namespace mo
