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
        MetaParams::instantiatePOD(table);
        MetaParams::instantiateVectors(table);
        MetaParams::instCV(table);
        MetaParams::instCVVec(table);
        MetaParams::instCVRect(table);
        MetaParams::instMOTypes(table);
    }
}
