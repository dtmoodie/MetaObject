#pragma once
#include "Export.hpp"
#include <string>

namespace mo {
enum ParamFlags {
    None_e = 0,
    /* This flag is set if the Param is an input Param */
    Input_e = 1,
    /* This flag is set if the Param is an output parmaeter */
    Output_e = 2,
    /* This flag is set if hte Param is an indicator of the underlying state of an object
       thus it is read only access*/
    State_e = 4,
    /* This flag is set if the Param is a control input*/
    Control_e = 8,
    /* This flag is set if the Param's underlying type is a buffer object */
    Buffer_e = 16,
    /* This flag is set if the Param is an optional input */
    Optional_e = 32,
    /* Set this flag on an input Param to allow desychronization between it and
       other input Params */
    Desynced_e = 64,
    /* If this flag is set, the timestamp will not be set on this Param
     This is needed to differentiate between a Param that has not been set
     yet and one that will never be set */
    Unstamped_e = 128,
    /* Set this flag to signify that this Param should be the one used
       for synchronizing inputs. */
    Sync_e = 256,
    RequestBuffered_e = 512,
    Source_e = 1024,
    OwnsMutex_e = 65536 // Interally set by Params to determine if mutex needs to be deleted or not
};

enum UpdateFlags{
    ValueUpdated_e,
    InputSet_e,
    InputCleared_e,
    InputUpdated_e
};
MO_EXPORTS std::string paramFlagsToString(ParamFlags flags);
MO_EXPORTS ParamFlags stringToParamFlags(const std::string& str);

enum ParamType {
    TParam_e = 0,
    CircularBuffer_e,
    ConstMap_e,
    Map_e,
    StreamBuffer_e,
    BlockingStreamBuffer_e,
    NNStreamBuffer_e,
    Queue_e,
    BlockingQueue_e,
    DroppingQueue_e,
    ForceBufferedConnection_e = 1024,
    ForceDirectConnection_e = 2048
};
MO_EXPORTS std::string paramTypeToString(ParamType type);
MO_EXPORTS ParamType stringToParamType(const std::string& str);

}
