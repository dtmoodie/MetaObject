#pragma once
#include <MetaObject/detail/Export.hpp>
#include <bitset>
#include <string>
namespace mo
{
    class Context;
    template <typename T>
    struct EnumTraits;

    template <typename T>
    class EnumClassBitset
    {
      public:
        typedef typename std::underlying_type<T>::type UnderlyingType;
        EnumClassBitset()
            : c()
        {
        }
        EnumClassBitset(UnderlyingType v)
            : c(v)
        {
        }
        EnumClassBitset(T v)
            : c()
        {
            c.set(get_value(v));
        }

        bool test(T pos) const
        {
            return c.test(get_value(pos));
        }

        EnumClassBitset& reset(T pos)
        {
            c.reset(get_value(pos));
            return *this;
        }

        EnumClassBitset& set(T pos)
        {
            c.set(get_value(pos));
            return *this;
        }

        EnumClassBitset& flip(T pos)
        {
            c.flip(get_value(pos));
            return *this;
        }

      private:
        std::bitset<static_cast<UnderlyingType>(EnumTraits<T>::max)> c;
        typename std::underlying_type<T>::type get_value(T v) const
        {
            return static_cast<UnderlyingType>(v);
        }
    };

    enum class ParamFlags
    {
        None_e,
        /* This flag is set if the Param is an input Param */
        Input_e,
        /* This flag is set if the Param is an output parmaeter */
        Output_e,
        /* This flag is set if hte Param is an indicator of the underlying state of an object
           thus it is read only access*/
        State_e,
        /* This flag is set if the Param is a control input*/
        Control_e,
        /* This flag is set if the Param's underlying type is a buffer object */
        Buffer_e,
        /* This flag is set if the Param is an optional input */
        Optional_e,
        /* Set this flag on an input Param to allow desychronization between it and
           other input Params */
        Desynced_e,
        /* If this flag is set, the timestamp will not be set on this Param
         This is needed to differentiate between a Param that has not been set
         yet and one that will never be set */
        Unstamped_e,
        /* Set this flag to signify that this Param should be the one used
           for synchronizing inputs. */
        Sync_e,
        RequestBuffered_e,
        Source_e,
        Dynamic_e,   // Dynamically created parameter object
        OwnsMutex_e, // Interally set by Params to determine if mutex needs to be deleted or not
        Max_e
    };

    template <>
    struct EnumTraits<ParamFlags>
    {
        static const ParamFlags max = ParamFlags::Max_e;
    };

    enum UpdateFlags
    {
        ValueUpdated_e,
        InputSet_e,
        InputCleared_e,
        InputUpdated_e,
        BufferUpdated_e
    };

    MO_EXPORTS std::string paramFlagsToString(EnumClassBitset<ParamFlags> flags);
    MO_EXPORTS EnumClassBitset<ParamFlags> stringToParamFlags(const std::string& str);

    enum BufferFlags
    {
        DEFAULT,
        CIRCULAR_BUFFER,
        MAP_BUFFER,
        STREAM_BUFFER,
        BLOCKING_STREAM_BUFFER,
        DROPPING_STREAM_BUFFER,
        NEAREST_NEIGHBOR_BUFFER,
        QUEUE_BUFFER,
        BLOCKING_QUEUE_BUFFER,
        DROPPING_QUEUE_BUFFER,

        FORCE_BUFFERED = 1024,
        FORCE_DIRECT = 2048
    };

    MO_EXPORTS std::string BufferFlagsToString(BufferFlags type);
    MO_EXPORTS BufferFlags stringToBufferFlags(const std::string& str);

    MO_EXPORTS BufferFlags getDefaultBufferType(const Context* source, const Context* dest);
    MO_EXPORTS void setDefaultBufferType(const Context* source, const Context* dest, BufferFlags type);
}
