#pragma once
#include <boost/optional.hpp>
#include <type_traits>
#include <memory>
#include <MetaObject/logging/Log.hpp>
namespace mo {
template<class Type, class Enable = void> struct ParamTraits {};
template<class T1, class T2> struct LargerSize{
    static const bool value = sizeof(T1) > sizeof(T2);
};

template<class Type> struct ParamTraits<Type, typename std::enable_if<std::is_pod<Type>::value && LargerSize<Type, void*>::value>::type> {
    enum {
        REQUIRES_GPU_SYNC = 0,
        HAS_TRIVIAL_MOVE = 1
    };

    // What is stored in buffers / Params
    typedef Type Storage_t;
    typedef Type ConstStorage_t;
    // what is passed around within a thread
    typedef Type ConstStorageRef_t;
    // The type stored by TInputParam's
    typedef boost::optional<Type> InputStorage_t;
    // What is used as an input to an object
    typedef const Type* Input_t;

    static inline Storage_t copy(const Type& value) {
        return value;
    }

    template<class...Args>
    static Type& reset(Storage_t& input_storage, Args...args) {
        input_storage = Type(std::forward(args)...);
        return input_storage;
    }

    template<class...Args>
    static void reset(InputStorage_t& input_storage, Args...args) {
        input_storage = Type(std::forward(args)...);
    }
    template<class...Args>
    static void nullify(InputStorage_t& input_storage) {
        input_storage.reset();
    }
    static inline Type& getMutable(Storage_t& value){
        return value;
    }
    static inline const Type& get(ConstStorageRef_t value){
        return value;
    }
};

template<class Type> struct ParamTraits<Type, typename std::enable_if<std::is_pod<Type>::value && !LargerSize<Type, void*>::value>::type> {
    enum {
        REQUIRES_GPU_SYNC = 0,
        HAS_TRIVIAL_MOVE = 1
    };

    typedef Type Storage_t;
    typedef Type ConstStorage_t;
    typedef const Type& ConstStorageRef_t;
    typedef boost::optional<Type> InputStorage_t;
    typedef const Type* Input_t;


    static inline Storage_t copy(const Type& value){
        return value;
    }
    template<class...Args>
    static Type& reset(Storage_t& input_storage, Args&&...args) {
        input_storage = Type(std::forward<Args>(args)...);
        return input_storage;
    }

    template<class...Args>
    static void reset(InputStorage_t& input_storage, Args...args) {
        input_storage = Type(std::forward(args)...);
    }
    template<class...Args>
    static void nullify(InputStorage_t& input_storage) {
        input_storage.reset();
    }
    static inline Type& getMutable(Storage_t& value){
        return value;
    }
    static inline const Type& get(ConstStorageRef_t value){
        return value;
    }
};

template<class Type> struct ParamTraits<Type, typename std::enable_if<!std::is_pod<Type>::value>::type> {
    enum {
        REQUIRES_GPU_SYNC = 0,
        HAS_TRIVIAL_MOVE = 0
    };
    typedef std::shared_ptr<Type> Storage_t;
    typedef std::shared_ptr<const Type> ConstStorage_t;
    typedef const std::shared_ptr<Type>& ConstStorageRef_t;
    typedef std::shared_ptr<const Type> InputStorage_t;
    typedef const Type* Input_t;

    static inline Storage_t copy(const Type& value) {
        return Storage_t(new Type(value));
    }

    template<class...Args>
    static Type& reset(Storage_t& input_storage, Args&&...args) {
        input_storage.reset(new Type(std::forward<Args>(args)...));
        return *input_storage;
    }

    template<class...Args>
    static void reset(InputStorage_t& input_storage, Args...args) {
        input_storage = Type(std::forward(args)...);
    }
    template<class...Args>
    static void nullify(InputStorage_t& input_storage) {
        input_storage.reset();
    }
    static inline Type& getMutable(Storage_t& value){
        MO_ASSERT(value);
        return *value;
    }
    static inline const Type& get(ConstStorageRef_t value){
        MO_ASSERT(value);
        return *value;
    }
};
}
