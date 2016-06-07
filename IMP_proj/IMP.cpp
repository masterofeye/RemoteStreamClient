#include "IMP.h"

#include "ENC_CudaInterop.hpp"

namespace RW
{
    namespace IMP
    {
        void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
        {
            switch (SubModuleType)
            {
            case CORE::tenSubModule::nenEncode_NVIDIA:
            {
                RW::ENC::tstMyControlStruct *data = static_cast<RW::ENC::tstMyControlStruct*>(*Data);
                data->pcuYUVArray = *(this->pcOutput->_pcuArray);
                break;
            }
            case CORE::tenSubModule::nenGraphic_Color:
            {
                IMP::tstMyControlStruct *data = static_cast<IMP::tstMyControlStruct*>(*Data);
                data->pcInput->_pgMat = pcOutput->_pgMat;
                break;
            }
            default:
                break;
            }

        }
    }
}