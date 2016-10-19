#pragma once

#include "..\RWPluginInterface\Config.h"

#define SERVER
#ifdef SERVER
#define ENC_INTEL
//#define ENC_NVENC

//#define IMP_CROP
#ifdef IMP_CROP
//#define IMP_MERGE
#endif
#endif

