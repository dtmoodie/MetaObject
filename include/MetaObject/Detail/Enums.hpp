#pragma once

namespace mo
{
    enum ParameterType
    {
        None_e = 0,
        Input_e = 1,
        Output_e = 2,
        State_e = 4,
        Control_e = 8,
        Buffer_e = 16
    };
    enum ParameterTypeFlags
    {
        TypedParameter_e = 0,
        cbuffer_e ,
        cmap_e,
        map_e,
        StreamBuffer_e
    };
}