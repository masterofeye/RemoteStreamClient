#ifndef __RWVX_H__
#define __RWVX_H__


#include <VX/vx.h>

enum nvx_type_e {
    RWVX_TYPE_CALLBACK = VX_TYPE_VENDOR_STRUCT_START,     /**< \brief A \ref  */
    RWVX_TYPE_STRUCT_MAX,                                /**< \brief A floating value for comparison between structs and objects */

    RWVX_TYPE_OBJECT_MAX = VX_TYPE_VENDOR_OBJECT_START,  /**< \brief A floating value used for bound checking the RemoteWorkStation Framework object types. */
};


#define RWVX_LIBRARY_SCHLEISSHEIMER (0x01)

enum rwvw_kernel_e {
    RWVX_KERNEL_VIDEOGRAB = VX_KERNEL_BASE(VX_ID_FGS, RWVX_LIBRARY_SCHLEISSHEIMER)
};

/**
* \ingroup nvx_framework_basic_types
* \brief The extended set of kernels provided by Schleissheimer GmbH.
*/
#define NVX_LIBRARY_FGS (0x0)


/*
* \ingroup nvx_framework_basic_types
* \brief Defines a list of extended vision kernels.
*/
enum nvx_kernel_e {
    /*! \brief Specifies Harris Track Kernel. */
    NVX_KERNEL_TEST = VX_KERNEL_BASE(VX_ID_FGS, NVX_LIBRARY_FGS),
};
#endif