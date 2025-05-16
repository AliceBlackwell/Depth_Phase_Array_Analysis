/*
 * iscloc.h
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
#ifndef ISCLOC_H
#define ISCLOC_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
/*
 * PostgreSQL library
 */
#ifdef WITH_DB
#include <libpq-fe.h>
#endif
/*
 * Grand Central Dispatch
 */
#ifdef WITH_GCD
#include <dispatch/dispatch.h>
#else
#define SERIAL
#endif
/*
 * Lapack
 */
#ifdef MACOSX
#include <Accelerate/Accelerate.h>
#endif

/*
 * String lengths
 */
#define LINLEN 1024
#define PARLEN 30
#define VALLEN 255
#define FILENAMELEN 255
#define AGLEN 17
#define NETLEN 7
#define STALEN 7
#define PHALEN 9
#define ERRLEN 160
#define MAXBUF 1024
/*
 * tolerance values
 */
#define NULLVAL 9999999                                       /* null value */
#define SAMETIME_TOL 0.1              /* time tolerance for duplicate picks */
#define DEPSILON 1.e-8               /* for testing floating point equality */
#define CONV_TOL 1.e-8                             /* convergence tolerance */
#define ZERO_TOL 1.e-10                                   /* zero tolerance */
/*
 * limits (array sizes)
 */
#define MAXOPT 100                   /* max number of possible instructions */
#define MAXHYP 50                 /* max number of hypocenters for an event */
#define MAXPHA 5000                      /* max number of associated phases */
#define MAXAMP 10          /* max number of reported amplitudes for a phase */
#define MAXPHACODES 400         /* max number of IASPEI phase name mappings */
#define MAXNUMPHA 200                   /* max number of IASPEI phase names */
#define MAXTTPHA 95                   /* max number of phases with TT table */
#define PHA_PER_READ 80                /* max number of phases in a reading */
/*
 *
 * Array sizes for spline interpolation routines
 *
 */
#define DELTA_SAMPLES 6      /* max number of TT samples in delta direction */
#define DEPTH_SAMPLES 4      /* max number of TT samples in depth direction */
#define MIN_SAMPLES 2            /* min number of samples for interpolation */
/*
 *
 * Array sizes for neighbourhood algorithm routines
 *
 */
#define NA_MAXND       4                  /* max number of model parameters */
#define NA_MAXITER   100                        /* max number of iterations */
#define NA_MAXSAMP  5000                /* max number of models in a sample */
#define NA_MAXBIT     30       /* max direction numbers for Sobol sequences */
#define NA_MAXDEG     10                  /* max degree for SAS polynomials */
#define NA_MAXSEQ  (NA_MAXND * NA_MAXSAMP)    /* max quasi-random sequences */
#define NA_MAXMOD  (NA_MAXSAMP * NA_MAXITER + 1)    /* max number of models */
/*
 * degree <-> rad conversions
 */
#define PI    M_PI                                           /* value of pi */
#define PI2   M_PI_2                                       /* value of pi/2 */
#define TWOPI 2 * PI                                        /* value of 2pi */
#define EARTH_RADIUS 6371.                                /* Earth's radius */
#define RAD_TO_DEG (180./PI)                   /* radian - degree transform */
#define DEG_TO_RAD (PI/180.)                   /* degree - radian transform */
#define DEG2KM (DEG_TO_RAD * EARTH_RADIUS)       /* degrees to km transform */
/*
 * WGS84 ellipsoid
 */
#define FLATTENING     0.00335281066474                    /* f = 1/298.257 */

#define TRUE 1                                             /* logical true  */
#define FALSE 0                                            /* logical false */

#ifndef _FUNCS_
#define _FUNCS_
#define mabs(A)    ((A)<0 ? -(A):(A))         /* returns the abs value of A */
#define signum(A)  ((A)<0 ? -1 : 1)                /* returns the sign of A */
#define max(A,B)   ((A)>(B) ? (A):(B))        /* returns the max of A and B */
#define min(A,B)   ((A)<(B) ? (A):(B))        /* returns the min of A and B */
#define Sqrt(A)    ((A)>0 ? sqrt((A)):0.)               /* safe square root */
#define streq(A,B)  (strcmp ((A),(B)) == 0)    /* are two strings the same? */
#define swap(A,B)   { temp=(A);(A)=(B);(B)=temp; }           /* swap values */
#define swapi(A,B) { itemp=(A);(A)=(B);(B)=itemp; }          /* swap values */
#define sign(A,B)  ((B) < 0 ? ((A) < 0 ? (A) : -(A)) : ((A) < 0 ? -(A) : (A)))
#endif


/*
 * timing functionto measure execution times
 */
double secs(struct timeval *t0);

/*
 *
 * C structure definitions
 *
 */

/*
 *
 * Event structure
 *
 */
typedef struct event_rec {
    char eventid[AGLEN];                                /* event identifier */
    int evid;                                        /* event id (isc_evid) */
    int prime;                                   /* hypid of prime solution */
    int isc_hypid;        /* for a new event not filled until location done */
    int outdbprime;                     /* hypid of prime solution in outdb */
    int outdbisc_hypid;                               /* isc_hypid in outdb */
    char etype[5];                    /* event type ([dfklsu ][eihlmnqrx ]) */
    int numhyps;                      /* number of hypocenters to this evid */
    int numphas;                             /* number of associated phases */
    int numsta;                            /* number of associated stations */
    int numrd;                             /* number of associated readings */
    char depth_agency[AGLEN];                   /* agency for initial depth */
    char location_agency[AGLEN];             /* agency for initial location */
    char time_agency[AGLEN];              /* agency for initial origin time */
    char hypo_agency[AGLEN];                 /* agency for fixed hypocenter */
    double start_depth;                      /* Value to start at or fix to */
    double start_lat;                        /* Value to start at or fix to */
    double start_lon;                        /* Value to start at or fix to */
    double start_time;                       /* Value to start at or fix to */
    int surface_fix;  /* Event occured at surface, don't try options 2 or 3 */
    int depth_fix;                              /* Flag that depth is fixed */
    int depth_fix_editor;             /* Flag that depth is fixed by editor */
    int fix_depth_depdp;   /* Instruction to calculate depdp then fix on it */
    int fix_depth_default;                 /* Fix to regional default depth */
    int fix_depth_median;               /* Fix to median of reported depths */
    int time_fix;                         /* Flag that origin time is fixed */
    int epi_fix;                            /* Flag that epicentre is fixed */
    int hypo_fix;                          /* Flag that EVERYTHING is fixed */
    int numagency;       /* number of agencies contributing arrivals to ISC */
} EVREC;
/*
 *
 * Hypocenter structure
 *
 */
typedef struct hyp_rec {
    int hypid;                                             /* hypocenter id */
    char origid[15];                                   /* origin identifier */
    double time;                            /* hypocenter origin epoch time */
    double lat;                           /* hypocenter geographic latitude */
    double lon;                          /* hypocenter geographic longitude */
    double depth;                                  /* hypocenter depth [km] */
    int nsta;                  /* number of readings NOT number of stations */
    int ndefsta;                             /* number of defining stations */
    int nass;                                /* number of associated phases */
    int ndef;                                  /* number of defining phases */
    double mindist;         /* distance to closest station (entire network) */
    double maxdist;        /* distance to furthest station (entire network) */
    double azimgap;                       /* azimuthal gap (entire network) */
    double sgap;                /* secondary azimuthal gap (entire network) */
    char etype[5];                    /* event type ([dfklsu ][eihlmnqrx ]) */
    char agency[AGLEN];                                      /* agency code */
    char rep[AGLEN];                                       /* reporter code */
    double sdobs;                               /* sigma * sqrt(N / (N -M)) */
    double stime;                                      /* origin time error */
    double sdepth;                                      /* depth time error */
    double smajax;                    /* error ellipse semi-major axis [km] */
    double sminax;                    /* error ellipse semi-minor axis [km] */
    double strike;                                  /* error ellipse strike */
    double depdp;                /* hypocenter depth from depth phases [km] */
    double dpderr;         /* hypocenter depth error from depth phases [km] */
    int depfix;                                              /* fixed depth */
    int epifix;                                          /* fixed epicenter */
    int timfix;                                        /* fixed origin time */
    int rank;                                        /* order of preference */
    int ignorehypo;             /* do not use in setting initial hypocentre */
} HYPREC;
/*
 *
 * Solution structure
 *
 */
typedef struct sol_rec {
    int converged;                              /* convergent solution flag */
    int diverging;                               /* divergent solution flag */
    int number_of_unknowns;      /* number of model parameters to solve for */
    int numphas;                       /* total number of associated phases */
    int hypid;                                             /* hypocenter id */
    double time;                            /* hypocenter origin epoch time */
    double lat;                           /* hypocenter geographic latitude */
    double lon;                          /* hypocenter geographic longitude */
    double depth;                                  /* hypocenter depth [km] */
    double depdp;                /* hypocenter depth from depth phases [km] */
    int ndp;      /* number of depth phases used in depth from depth phases */
    double depdp_error;    /* hypocenter depth error from depth phases [km] */
    double urms;                             /* unweighted RMS residual [s] */
    double wrms;                               /* weighted RMS residual [s] */
    double covar[4][4];                          /* model covariance matrix */
    double error[4];                                       /* uncertainties */
    double smajax;                    /* error ellipse semi-major axis [km] */
    double sminax;                    /* error ellipse semi-minor axis [km] */
    double strike;                                  /* error ellipse strike */
    double sdobs;                                /* urms * sqrt(N / (N -M)) */
    int nass;                                /* number of associated phases */
    int ndef;                                  /* number of defining phases */
    int prank;                            /* rank of data covariance matrix */
    int nreading;                                     /* number of readings */
    int ndefsta;                             /* number of defining stations */
    double mindist;         /* distance to closest station (entire network) */
    double maxdist;        /* distance to furthest station (entire network) */
    double azimgap;                       /* azimuthal gap (entire network) */
    double sgap;                /* secondary azimuthal gap (entire network) */
    double bodymag;                                               /* ISC mb */
    double surfmag;                                               /* ISC Ms */
    double bodymag_uncertainty;                        /* mb standard error */
    double surfmag_uncertainty;                        /* Ms standard error */
    int nsta_mb;         /* number of station magnitudes used in network mb */
    int nsta_ms;         /* number of station magnitudes used in network Ms */
    int nass_mb;             /* number of readings associated to network mb */
    int nass_ms;             /* number of readings associated to network Ms */
    int mb_id;                                   /* mb magnitude identifier */
    int ms_id;                                   /* Ms magnitude identifier */
    int nmbagency;        /* number of agencies contributing amps to ISC mb */
    int nMsagency;        /* number of agencies contributing amps to ISC Ms */
    int epifix;                                          /* fixed epicenter */
    int timfix;                                        /* fixed origin time */
    int depfix;                                              /* fixed depth */
    int depfixtype;                             /* fixed depth type [0..10] */
} SOLREC;
/*
 *
 * Network quality metrics structure
 *
 */
typedef struct network_qual {
    double gap;                             /* primary azimuthal gap [deg] */
    double sgap;                          /* secondary azimuthal gap [deg] */
    double du;                                   /* network quality metric */
    int ndefsta;                            /* number of defining stations */
    int ndef;                                 /* number of defining phases */
    double mindist;                                      /* min dist [deg] */
    double maxdist;                                      /* max dist [deg] */
} NETQUAL;
/*
 *
 * Hypocenter quality structure
 *
 */
typedef struct hypo_qual {
    int hypid;                                             /* hypocenter id */
    int ndefsta_10km;           /* number of defining stations within 10 km */
    int gtcand;                          /* 1 if GT5 candidate, 0 otherwise */
    NETQUAL local_net;                  /* local (0-150 km) network quality */
    NETQUAL near_net;           /* near-regional (3-10 deg) network quality */
    NETQUAL tele_net;           /* teleseismic (28-180 deg) network quality */
    NETQUAL whole_net;                 /* whole (0-180 deg) network quality */
    int ndepthphase;                     /* number of defining depth phases */
    double score;                          /* event score for ranking hypos */
} HYPQUAL;
/*
 *
 * Magnitude attenuation Q(d,h) structure
 *
 */
typedef struct mag_atten {
	int	num_dists;                            /* number of distance samples */
	int	num_depths;                              /* number of depth samples */
    double mindist;                                      /* min delta [deg] */
    double maxdist;                                      /* max delta [deg] */
    double mindepth;                                      /* min depth [km] */
    double maxdepth;                                      /* max depth [km] */
    double *deltas;                               /* distance samples [deg] */
    double *depths;                                   /* depth samples [km] */
    double **q;                                                   /* Q(d,h) */
} MAGQ;
/*
 *
 * Station magnitude structure
 *
 */
typedef struct stamag_rec {
    char sta[STALEN];                                       /* station code */
    char prista[STALEN];                /* to deal with alternate sta codes */
    char agency[AGLEN];                        /* agency (station operator) */
    char deploy[AGLEN];                             /* deployment (network) */
    char lcn[3];                                                /* location */
    double magnitude;                                          /* magnitude */
    int magdef;                    /* 1 if used for the netmag, 0 otherwise */
    int mtypeid;                                       /* magnitude type id */
    char magtype[PHALEN];                                 /* magnitude type */
} STAMAG;
/*
 *
 * Reading magnitude structure
 *
 */
typedef struct rdmag_rec {
    int rdid;                                                 /* reading id */
    char sta[STALEN];                                       /* station code */
    char prista[STALEN];                /* to deal with alternate sta codes */
    char agency[AGLEN];                        /* agency (station operator) */
    char deploy[AGLEN];                             /* deployment (network) */
    char lcn[3];                                                /* location */
    double magnitude;                                          /* magnitude */
    int magdef;                    /* 1 if used for the stamag, 0 otherwise */
    int mtypeid;                                       /* magnitude type id */
    char magtype[PHALEN];                                 /* magnitude type */
} RDMAG;
/*
 *
 * MS vertical/horizontal magnitude structure
 *
 */
typedef struct mszh_rec {
    int rdid;                                                 /* reading id */
    double msz;                                              /* vertical MS */
    int mszdef;                    /* 1 if used for the stamag, 0 otherwise */
    double msh;                                            /* horizontal MS */
    int mshdef;                    /* 1 if used for the stamag, 0 otherwise */
} MSZH;
/*
 *
 * Amplitude structure
 *
 */
typedef struct amp_rec {
    int ampid;                                              /* amplitude id */
    int magid;                                      /* network magnitude id */
    char amplitudeid[255];                          /* amplitude identifier */
    double amp;                              /* peak-to-peak amplitude [nm] */
    double per;                                               /* period [s] */
    double logat;                                               /* log(A/T) */
    double snr;                                    /* signal-to-noise ratio */
    char ach[5];                                  /* amplitude channel code */
    char comp;                                                 /* component */
    int ampdef;               /* 1 if used for the reading mag, 0 otherwise */
    double magnitude;                                          /* magnitude */
    int mtypeid;                                       /* magnitude type id */
    char magtype[PHALEN];                                 /* magnitude type */
} AMPREC;
/*
 *
 * Phase structure
 *
 */
typedef struct pha_rec {
    int hypid;                                             /* hypocenter id */
    int phid;                                                   /* phase id */
    int rdid;                                                 /* reading id */
    int init;                           /* initial phase in a reading [0/1] */
    int repid;                            /* report id from association row */
    char arrid[15];                                   /* arrival identifier */
    char pickid[255];                                    /* pick identifier */
    char auth[AGLEN];                                           /* observer */
    char rep[AGLEN];                                            /* reporter */
    char rep_phase[PHALEN];                               /* reported phase */
    char phase[PHALEN];                                        /* ISC phase */
    char fdsn[30];        /* fdsn station code: agency.sta.network.location */
    char sta[STALEN];                                       /* station code */
    char prista[STALEN];                /* to deal with alternate sta codes */
    char agency[AGLEN];                        /* agency (station operator) */
    char deploy[AGLEN];                             /* deployment (network) */
    char lcn[3];                                                /* location */
    double sta_lat;                               /* station latitude [deg] */
    double sta_lon;                              /* station longitude [deg] */
    double sta_elev;                                       /* elevation [m] */
    double time;                                  /* arrival epoch time [s] */
    double slow;                               /* measured slowness [s/deg] */
    double azim;                                  /* measured azimuth [deg] */
    double snr;                                    /* signal-to-noise ratio */
    double esaz;            /* event-to-station azimuth cf current solution */
    double seaz;            /* station-to-event azimuth cf current solution */
    double delta;                              /* delta cf current solution */
    char pch[5];                                      /* phase channel code */
    char comp;                                                 /* component */
    char sp_fm;                                             /* first motion */
    char lp_fm;                           /* long period first motion [cd ] */
    char detchar;                              /* detection character [ei ] */
    int phase_fixed;              /* 1 to stop iscloc reidentifying a phase */
    int timedef;                  /* 1 if used in the location, 0 otherwise */
    int force_undef; /* 1 if forced to be undefining by editor, 0 otherwise */
    int firstP;              /* 1 if first arriving defining P, 0 otherwise */
    int firstS;                       /* 1 if first arriving S, 0 otherwise */
    double ttime;                       /* travel time with corrections [s] */
    double dtdd;                             /* horizontal slowness [s/deg] */
    double dtdh;                                /* vertical slowness [s/km] */
    double bpdel;                /* depth phase bounce point distance [deg] */
    double resid;                                      /* time residual [s] */
    int hasdepthphase;        /* 1 if firstP and reading has depth phase(s) */
    int pPindex;                     /* index pointer to pP in this reading */
    int pwPindex;                   /* index pointer to pwP in this reading */
    int pSindex;                     /* index pointer to pS in this reading */
    int sPindex;                     /* index pointer to sP in this reading */
    int sSindex;                     /* index pointer to sS in this reading */
    char prevphase[PHALEN];            /* ISC phase from previous iteration */
    int prevtimedef;                     /* timedef from previous iteration */
    double measerr;              /* a priori measurement error estimate [s] */
    double dupsigma;            /* extra variance factor for duplicates [s] */
    int duplicate;                           /* 1 if duplicate, 0 otherwise */
    int covindex;                          /* position in covariance matrix */
    int numamps;                           /* number of reported amplitudes */
    int purged;                                    /* used by iscloc_search */
    AMPREC a[MAXAMP];                                  /* amplitude records */
} PHAREC;
/*
 *
 * Reading structure (phase indices belonging to a reading)
 *
 */
typedef struct rdidx {
    int start;                          /* start phase index in the reading */
    int npha;                            /* number of phases in the reading */
} READING;
/*
 *
 * Phase map structure (map reported phase codes to IASPEI phase codes)
 *
 */
typedef struct phase_map_rec {
    char rep_phase[PHALEN];                               /* reported phase */
    char phase[PHALEN];                                     /* IASPEI phase */
} PHASEMAP;
/*
 *
 * A priori measurement error estimates structure
 *
 */
typedef struct phaseweight {
    char phase[PHALEN];                                       /* phase name */
    double delta1;                                    /* min distance [deg] */
    double delta2;                                    /* max distance [deg] */
    double measurement_error;    /* a priori measurement error estimate [s] */
} PHASEWEIGHT;
/*
 *
 * Station info structure
 *
 */
typedef struct sta_rec {
    char fdsn[30];        /* fdsn station code: agency.sta.network.location */
    char sta[STALEN];                                       /* station code */
    char altsta[STALEN];                        /* alternative station code */
    char agency[AGLEN];                        /* agency (station operator) */
    char deploy[AGLEN];                             /* deployment (network) */
    char lcn[3];                                                /* location */
    double lat;                                                 /* latitude */
    double lon;                                                /* longitude */
    double elev;                                               /* elevation */
} STAREC;
/*
 *
 * travel time table structure
 *     TT tables are generated using libtau software
 *
 */
typedef struct tt_tables {
    char phase[PHALEN];                                            /* phase */
	int	ndel;                                 /* number of distance samples */
	int	ndep;                                    /* number of depth samples */
    double *depths;                                   /* depth samples [km] */
    double *deltas;                               /* distance samples [deg] */
    double **tt;                                   /* travel-time table [s] */
    double **bpdel;        /* depth phase bounce point distance table [deg] */
    double **dtdd;                     /* horizontal slowness table [s/deg] */
    double **dtdh;                        /* vertical slowness table [s/km] */
} TT_TABLE;
/*
 *
 * ak135 ellipticity correction coefficients structure
 *     Note: the tau corrections are stored at 5 degree intervals in distance
 *           and at the depths 0, 100, 200, 300, 500, 700 km.
 *
 */
typedef struct ec_coef {
    char phase[PHALEN];                                            /* phase */
	int	num_dists;                            /* number of distance samples */
	int	num_depths;                              /* number of depth samples */
    double mindist;                               /* minimum distance [deg] */
    double maxdist;                               /* maximum distance [deg] */
    double depth[6];                                  /* depth samples [km] */
    double *delta;                                /* distance samples [deg] */
    double **t0;                                         /* t0 coefficients */
    double **t1;                                         /* t1 coefficients */
    double **t2;                                         /* t2 coefficients */
} EC_COEF;
/*
 *
 * Flinn-Engdahl geographic region numbers (1995)
 *
 */
typedef struct FlinnEngdahl {
	int	nlat;                                 /* number of latitude samples */
	int	*nl;                /* number of longitude samples at each latitude */
    int **lon;                         /* longitude ranges at each latitude */
    int **grn;                  /* grn in longitude ranges at each latitude */
} FE;
/*
 *
 * NA search space
 *
 */
typedef struct na_searchspace {
    int nd;                                   /* number of model parameters */
    int epifix, otfix, depfix;                     /* fixed parameter flags */
    double lat, lon, ot, depth;                   /* center point of search */
    double lpnorm;         /* p = [1,2] for L1, L2 norm or anything between */
    double range[NA_MAXND][2];                       /* search space limits */
    double ranget[NA_MAXND][2];           /* normalized search space limits */
    double scale[NA_MAXND+1];                                    /* scaling */
} NASPACE;
/*
 *
 * Sobol-Antonov-Saleev coefficients for quasi-random sequences
 *
 */
typedef struct sobol_coeff {
    int n;                           /* max number of independent sequences */
    int mdeg[NA_MAXSEQ];                                          /* degree */
    unsigned long pol[NA_MAXSEQ];                             /* polynomial */
    unsigned long iv[NA_MAXSEQ][NA_MAXBIT];          /* initializing values */
} SOBOL;
/*
 *
 * Variogram structure
 *
 */
typedef struct variogram {
    int n;                                             /* number of samples */
    double sill;                        /* sill (background variance) [s^2] */
    double maxsep;          /* max station separation to be considered [km] */
    double *x;                                  /* station separations [km] */
    double *y;                                    /* variogram values [s^2] */
    double *d2y;     /* second derivatives for natural spline interpolation */
} VARIOGRAM;
/*
 *
 * Phaselist structure for defining phases
 *
 */
typedef struct phaselst {
    char phase[PHALEN];                                       /* phase name */
    int n;                                             /* number of samples */
    int *ind;   /* permutation vector to block-diagonalize data covariances */
} PHASELIST;

/*
 *
 * Nearest-neighbour station order
 *
 */
typedef struct staord {
    int index;
    int x;
} STAORDER;

/*
 *
 * node stucture from single-linkage clustering
 *
 */
typedef struct node {
    int left;
    int right;
    double linkdist;
} NODE;


/*
 *
 * function declarations
 *
 */

/*
 * calc_ellipticity_corr.c
 */
EC_COEF *read_elcor_tbl(int *num_phases, char *filename);
void free_elcor_tbl(EC_COEF *ec, int num_phases);
double get_ellip_corr(EC_COEF *ec, char * phase, double ecolat,
            double delta, double depth, double esaz);
/*
 * calc_uncertainty.c
 */
int calc_error(SOLREC *sp, PHAREC p[]);
/*
 * cluster.c
 */
int HierarchicalCluster(int nsta, double **distmatrix, STAORDER staorder[]);
/*
 * data_covariance.c
 */
STAREC *get_stalist(int numphas, PHAREC p[], int *nsta);
int starec_compare(const void *sta1, const void *sta2);
double **distance_matrix(int nsta, STAREC stalist[]);
int get_sta_index(int nsta, STAREC stalist[], char *sta);
void sort_phaserec_nn(int numphas, int nsta, PHAREC p[], STAREC stalist[],
            STAORDER staorder[]);
double **data_covariance_matrix(int nsta, int numphas, int nd, PHAREC p[],
            STAREC stalist[], double **distmatrix, VARIOGRAM *variogramp);
int read_variogram(char *fname, VARIOGRAM *variogramp);
void free_variogram(VARIOGRAM *variogramp);
/*
 * depth_phases.c
 */
int depth_resolution(SOLREC *sp, READING *rdindx, PHAREC p[], int isverbose);
int depth_phase_check(SOLREC *sp, READING *rdindx, PHAREC p[], int isverbose);
int depth_phase_stack(SOLREC *sp, PHAREC p[], TT_TABLE *tt_tables,
            short int **topo);
/*
 * distaz.c
 */
void calc_delaz(SOLREC *sp, PHAREC p[], int verbosethistime);
double distaz(double slat, double slon, double elat, double elon,
            double *azi, double *baz, int isnew);
void deltaloc(double lat1, double lon1, double delta, double azim,
            double *lat2, double *lon2);
/*
 * gregion.c
 */
int read_FlinnEngdahl(char *fname, FE *fep);
void free_fe(FE *fep);
int gregnum(double lat, double lon, FE *fep);
int gregtosreg(int grn);
int gregion(int number, char *gregname);
int sregion(int number, char *sregname);
double **read_default_depth_grid(char *fname, double *gres, int *ngrid);
double *read_default_depth_region(char *fname);
double get_default_depth(SOLREC *sp, int ngrid, double gres,
            double **depthgrid, FE *fep, double *grn_depth, int *isdefdep);
/*
 * initializations.c
 */
int init_event(EVREC *ep, HYPREC h[]);
void start_hyp(EVREC *ep, HYPREC h[], HYPREC *starthyp);
int init_sol(SOLREC *sp, EVREC *ep, HYPREC *hp);
/*
 * interpolate.c
 */
void spline(int n, double *x, double *y, double *d2y, double *tmp);
double spline_int(double xp, int n, double *x, double *y, double *d2y,
            int isderiv, double *dydx, double *d2ydx);
void bracket(double xp, int n, double *x, int *jlo, int *jhi);
void ibracket(int xp, int n, int *x, int *jlo, int *jhi);
double bilinear_int(double xp1, double xp2, int nx1, int nx2,
            double *x1, double *x2, double **y);
/*
 * isf.c
 */
int read_isf(FILE *infile, int isf, EVREC *ep, HYPREC *hp[], PHAREC *pp[],
            STAREC stationlist[],STAMAG *stamag_mb[], STAMAG *stamag_ms[],
            RDMAG *rdmag_mb[], RDMAG *rdmag_ms[], MSZH *mszh[], char *magbloc);
void write_isf(FILE *ofp, EVREC *ep, SOLREC *sp, HYPREC h[], PHAREC p[],
            STAMAG *stamag_mb, STAMAG *stamag_ms,
            RDMAG rdmag_mb[], RDMAG rdmag_ms[], int grn, char *magbloc);
int read_stafile(STAREC *stationlistp[]);
int read_stafile_isf2(STAREC *stationlistp[]);
int check_int(char *substr);
void sort_phaserec_isf(int numphas, PHAREC p[]);
/*
 * locator_parallel.c
 */
int eventloc(int isf, int database, int *total, int *fail, int *opt,
            EVREC *e, HYPREC h[], SOLREC *s, PHAREC p[],
            STAMAG *stamag_mb, STAMAG *stamag_ms,
            RDMAG *rdmag_mb, RDMAG *rdmag_ms, MSZH *mszh, int ismbQ, MAGQ *mbQ,
            EC_COEF *ec, TT_TABLE *tt_tables, VARIOGRAM *variogram,
            double gres, int ngrid, double **depthgrid,
            FE *fe, double *grn_depth, short int **topo, FILE *isfout,
            char *magbloc);
void synthetic(EVREC *ep, HYPREC *hp, SOLREC *sp, READING *rdindx, PHAREC p[],
            EC_COEF *ec, TT_TABLE *tt_tables, short int **topo,
            int database, int isf);
void fixedhypo(EVREC *ep, HYPREC *hp, SOLREC *sp, READING *rdindx, PHAREC p[],
            EC_COEF *ec, TT_TABLE *tt_tables, short int **topo);
int locate_event(int option, int nsta, int has_depdpres, SOLREC *sp,
            READING *rdindx, PHAREC p[], EC_COEF *ec, TT_TABLE *tt_tables,
            STAREC stalist[], double **distmatrix, VARIOGRAM *variogramp,
            STAORDER staorder[], short int **topo);
int getphases(int numphas, PHAREC p[], PHASELIST plist[]);
void freephaselist(int nphases, PHASELIST plist[]);
void readings(int numphas, int nreading, PHAREC p[], READING *rdindx);
/*
 * loc_qual.c
 */
int location_quality(int numphas, PHAREC p[], HYPQUAL *hq);
double gapper(int nsta, double *esaz, double *gap, double *sgap);
/*
 * magnitude.c
 */
int calc_netmag(SOLREC *sp, READING *rdindx, PHAREC p[],
            STAMAG *stamag_mb, STAMAG *stamag_ms,
            RDMAG *rdmag_mb, RDMAG *rdmag_ms, MSZH *mszh,
            int ismbQ, MAGQ *mbQ);
int read_magQ(char *filename, MAGQ * magq);
/*
 * na_parallel.c
 */
int set_searchspace(SOLREC *sp, NASPACE *nasp);
int na_search(int nsta, SOLREC *sp, PHAREC p[],TT_TABLE *tt_tables,
            EC_COEF *ec, short int **topo, STAREC stalist[],
            double **distmatrix, VARIOGRAM *variogramp, STAORDER staorder[],
            NASPACE *nasp, char *filename);
/*
 * pgsql_funcs.c
 */
void pgsql_error(char *message);
int pgsql_conn(void);
void pgsql_disconn(void);
/*
 * phaseids_parallel.c
 */
int id_pha(SOLREC *sp, READING *rdindx, PHAREC p[], EC_COEF *ec,
            TT_TABLE *tt_tables, short int **topo);
int reidentify_pha(SOLREC *sp, READING *rdindx, PHAREC p[], EC_COEF *ec,
            TT_TABLE *tt_tables, short int **topo);
void id_pfake(SOLREC *sp, PHAREC p[], EC_COEF *ec, TT_TABLE *tt_tables,
            short int **topo);
void remove_pfake(SOLREC *sp, PHAREC p[]);
int mark_duplicates(SOLREC *sp, PHAREC p[], EC_COEF *ec, TT_TABLE *tt_tables,
            short int **topo);
void reported_phase_resid(SOLREC *sp, PHAREC *pp, EC_COEF *ec,
            TT_TABLE *tt_tables, short int **topo);
/*
 * print_event.c
 */
void print_sol(SOLREC *sp, int grn);
void print_hyp(EVREC *ep, HYPREC h[]);
void print_pha(int numphas, PHAREC p[]);
void print_defining_pha(int numphas, PHAREC p[]);
/*
 * read_db.c
 */
int get_data(EVREC *ep, HYPREC *hp[], PHAREC *pp[],
            STAMAG *stamag_mb[], STAMAG *stamag_ms[],
            RDMAG *rdmag_mb[], RDMAG *rdmag_ms[], MSZH *mszh[]);
void sort_phaserec_db(int numphas, PHAREC p[]);
/*
 * read_textfiles.c
 */
int read_config(char *filename);
int read_model(char *filename);
int read_instruction(char* instruction, EVREC *ep, int isf);
int read_data_files(char *configdir, int *ismbQ, MAGQ *mbQp, FE *fep,
            double **grn_depth, double *gres, int *ngrid, double ***depthgrid,
            short int ***topo, int *num_ecphases, EC_COEF *ec[],
            TT_TABLE *tt_tables[], VARIOGRAM *variogramp);
/*
 * svd_parallel.c
 */
int svd_decompose(int n, int m, double **u, double sv[], double **v);
int svd_solve(int n, int m, double **u, double sv[], double **v,
            double *b, double *x, double thres);
void svd_model_covariance_matrix(int m, double thres, double sv[], double **v,
            double mcov[][4]);
double svd_threshold(int n, int m, double sv[]);
int svd_rank(int n, int m, double sv[], double thres);
double svd_norm(int m, double sv[], double thres, double *cond);
int projection_matrix(int numphas, PHAREC p[], int n, double pctvar,
            double **cov, double **w, int *prank, int nunp,
            char **phundef, int ispchange);
/*
 * timefuncs.c
 */
double epoch_time(char *htime, int msec);
double epoch_time_isf(int yyyy, int mm, int dd, int hh, int mi, int ss,
            int msec);
void human_time(char *htime, double etime);
void human_time_isf(char *hday, char *htim, double etime);
double read_time(char *timestr);
/*
 * traveltimes_parallel.c
 */
TT_TABLE *read_tt_tables(char *dirname);
void free_tt_tbl(TT_TABLE *tt_tables);
short int **read_etopo1(char *filename);
int get_phase_index(char *phase);
int calc_resid(SOLREC *sp, PHAREC p[], char mode[4], EC_COEF *ec,
            TT_TABLE *tt_tables, short int **topo, int iszderiv);
int read_ttime(SOLREC *sp, PHAREC *pp, EC_COEF *ec, TT_TABLE *tt_tables,
            short int **topo, int iszderiv, int isfirst);
double get_tt(TT_TABLE *tt_tablep, double depth, double delta,
            int iszderiv, double *dtdd, double *dtdh, double *bpdel);
double topcor(int ips, double rayp, double bplat, double bplon,
            short int **topo, double *tcorw);
/*
 * utils.c
 */
void skipcomments(char *buf, FILE *fp);
char *rtrim(char *buf);
int dropspace(char *str1, char *str2);
void Free(void *ptr);
double **alloc_matrix(int nrow, int ncol);
void free_matrix(double **matrix);
short int **alloc_i2matrix(int nrow, int ncol);
void free_i2matrix(short int **matrix);
int int_compare(const void *x, const void *y);
int double_compare(const void *x, const void *y);
/*
 * write_db.c
 */
int put_data(EVREC *ep, SOLREC *sp, PHAREC p[], HYPQUAL *hq,
            STAMAG *stamag_mb, STAMAG *stamag_ms,
            RDMAG *rdmag_mb, RDMAG *rdmag_ms, MSZH *mszh, FE *fep);
void put_hypoc_err(EVREC *ep, SOLREC *sp);
void put_hypoc_acc(EVREC *ep, SOLREC *sp, HYPQUAL *hq);
int put_netmag(SOLREC *sp);
int put_stamag(SOLREC *sp, STAMAG *stamag_mb, STAMAG *stamag_ms);
int put_rdmag(SOLREC *sp, RDMAG *rdmag_mb, RDMAG *rdmag_ms);
int put_mszh(SOLREC *sp, MSZH *mszh);
int put_ampmag(SOLREC *sp, PHAREC p[]);
void put_assoc(SOLREC *sp, PHAREC p[]);
void remove_isc(EVREC *ep);
void replace_prime(EVREC *ep, HYPREC *hp);
void replace_assoc(EVREC *ep, PHAREC p[], HYPREC *hp);
int get_id(char *sequence, int *nextid);


#endif	/* ISCLOC_H */
