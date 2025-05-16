/*
 * rstt.h
 *    C structure and function declarations for the ISC location code
 *
 *  Istvan Bondar
 *  Research Centre for Astronomy and Earth Sciences,
 *  Hungarian Academy of Sciences
 *  Geodetic and Geophysical Institute,
 *  Kovesligethy Rado Seismological Observatory
 *  Meredek utca 18, Budapest, H-1112, Hungary
 *  bondar@seismology.hu
 *  ibondar2014@gmail.com
 *
 */
#ifndef RSTT_H
#define RSTT_H

#include "iscloc.h"
#define MAX_RSTT_DIST 15
/*
 * RSTT
 */
#ifndef WITH_RSTT
#define WITH_RSTT
#endif

#ifndef SLBM_H
#define SLBM_H
#endif

#ifndef SLBM_C_SHELL_H
#include "slbm_C_shell.h"
#define SLBM_C_SHELL_H
#endif

#endif /* RSTT_H */
