#ifndef RPPI_FILTERING_FUNCTIONS_H
#define RPPI_FILTERING_FUNCTIONS_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif
RppStatus
Rppi_median_8u_pkd1_host(const Rpp8u * srcPtr, int rSrcStep,Rpp8u * dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_median_8u_pkd1(const Rpp8u * srcPtr, int rSrcStep,Rpp8u * dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_gaussian_8u_pkd1_host(const Rpp8u * srcPtr, Rpp8u * dstPtr, RppiSize oROI, RppiAxis flip);
#ifdef __cplusplus
}
#endif
#endif /* RPPI_FILTERING_FUNCTIONS_H */