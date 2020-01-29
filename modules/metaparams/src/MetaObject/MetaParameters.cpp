#include "MetaParameters.hpp"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
struct SystemTable;
namespace mo
{
    namespace metaparams
    {
        void instantiatePOD(SystemTable* table);
        void instantiateVectors(SystemTable* table);
        void instCV(SystemTable* table);
        void instCVVec(SystemTable* table);
        void instCVRect(SystemTable* table);
        void instMOTypes(SystemTable* table);
    } // namespace metaparams
    void initMetaParamsModule(SystemTable* table)
    {
        PerModuleInterface::GetInstance()->SetSystemTable(table);
        metaparams::instantiatePOD(table);
        metaparams::instantiateVectors(table);
#ifdef MO_HAVE_OPENCV
        metaparams::instCV(table);
        metaparams::instCVVec(table);
        metaparams::instCVRect(table);
#endif
        metaparams::instMOTypes(table);
    }

} // namespace mo
extern "C" {
void initModuleWithSystemTable(SystemTable* table)
{
    mo::initMetaParamsModule(table);
}
}
