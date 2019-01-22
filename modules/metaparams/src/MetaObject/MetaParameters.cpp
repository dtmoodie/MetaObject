#include "MetaParameters.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
struct SystemTable;
namespace mo
{
    namespace MetaParams
    {
        void instantiatePOD(SystemTable* table);
        void instantiateVectors(SystemTable* table);
        void instCV(SystemTable* table);
        void instCVVec(SystemTable* table);
        void instCVRect(SystemTable* table);
        void instMOTypes(SystemTable* table);
    }
    void initMetaParamsModule(SystemTable* table)
    {
        PerModuleInterface::GetInstance()->SetSystemTable(table);
        mo::MetaParams::instantiatePOD(table);
        mo::MetaParams::instantiateVectors(table);
    #ifdef MO_HAVE_OPENCV
        mo::MetaParams::instCV(table);
        mo::MetaParams::instCVVec(table);
        mo::MetaParams::instCVRect(table);
    #endif
        mo::MetaParams::instMOTypes(table);
    }

}
extern "C"{
    void initModuleWithSystemTable(SystemTable* table)
    {
        mo::initMetaParamsModule(table);
    }
}
