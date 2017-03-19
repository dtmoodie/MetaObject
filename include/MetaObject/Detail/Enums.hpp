#pragma once
#include "Export.hpp"
#include <string>
namespace mo
{
    enum ParameterType
    {
        None_e = 0,
		/* This flag is set if the parameter is an input parameter */
        Input_e = 1,
		/* This flag is set if the parameter is an output parmaeter */
        Output_e = 2,
		/* This flag is set if hte parameter is an indicator of the underlying state of an object
		   thus it is read only access*/
        State_e = 4,
		/* This flag is set if the parameter is a control input*/
        Control_e = 8,
		/* This flag is set if the parameter's underlying type is a buffer object */
        Buffer_e = 16,
		/* This flag is set if the parameter is an optional input */
        Optional_e = 32,
		/* Set this flag on an input parameter to allow desychronization between it and
		   other input parameters */
        Desynced_e = 64,
		/* If this flag is set, the timestamp will not be set on this parameter
		 This is needed to differentiate between a parameter that has not been set
		 yet and one that will never be set */
        Unstamped_e = 128,
		/* Set this flag to signify that this parameter should be the one used
		   for synchronizing inputs. */
		Sync_e = 256
    };
    MO_EXPORTS std::string ParameteTypeToString(ParameterType type);
    MO_EXPORTS ParameterType StringToParameteType(const std::string& str);
    enum ParameterTypeFlags
    {
        TypedParameter_e = 0,
        CircularBuffer_e ,
        ConstMap_e,
        Map_e,
        StreamBuffer_e,
        BlockingStreamBuffer_e,
        NNStreamBuffer_e,
        Queue_e,
        ForceBufferedConnection_e = 1024,
        ForceDirectConnection_e = 2048
    };
    MO_EXPORTS std::string ParameterTypeFlagsToString(ParameterTypeFlags flags);
    MO_EXPORTS ParameterTypeFlags StringToParameterTypeFlags(const std::string& str);

}
