
#ifndef __H_NORM_INF__
#define __H_NORM_INF__

#include <core/datadefs.h>

void norm_inf( real *dev, real *host, int len, real *res, real *idx, real *scratch);
void norm_inf_host( real *dev, real *host, int len, real *res, real *idx, real *scratch);


#endif
