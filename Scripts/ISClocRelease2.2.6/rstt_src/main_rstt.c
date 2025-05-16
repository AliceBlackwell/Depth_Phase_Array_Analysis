/*
 *  iscloc_rstt2.2.6
 *
 *  Istvan Bondar
 *  International Seismological Centre
 *  Pipers Lane, Thatcham
 *  Berkshire, RG19 4NS
 *  United Kingdom
 *
 *  now at
 *  Research Centre for Astronomy and Earth Sciences,
 *  Hungarian Academy of Sciences
 *  Geodetic and Geophysical Institute,
 *  Kovesligethy Rado Seismological Observatory
 *  Meredek utca 18, Budapest, H-1112, Hungary
 *  bondar@seismology.hu
 *  ibondar2014@gmail.com
 *
 *
 *  ISC single-event location with RSTT travel-time predictions
 *      neighbourhood algorithm (NA) grid-search for a refined initial
 *          hypocentre guess (Sambridge, 1999; Sambridge and Kennett, 2001)
 *      iterative linearized least-squares inversion that accounts for
 *          correlated model errors (Bondar and McLaughlin, 2009)
 *      nearest-neighbour ordering of stations to block-diagonalize data
 *          covariance matrix (de Hoon et al., 2004)
 *      depth-phase stacking for robust depth estimation
 *          (Murphy and Barker, 2006)
 *      ellipticity (Dziewonski and Gilbert, 1976, Kennett and Gudmundsson,
 *          1996) and elevation corrections
 *      bounce point corrections for depth phases as well as water column
 *          correction for pwP (Engdahl et al., 1998)
 *      free-depth solution is attempted only if there is depth resolution
 *          (number of defining depth phases > mindepthpha   &&
 *             number of agencies reporting depth phases >= ndepagency) ||
 *           number of local defining stations >= minlocalsta           ||
 *           number of defining P and S pairs >= min_s_p                ||
 *           (number of defining PcP, ScS >= min_corepha &&
 *             number of agencies reporting core reflections >= ndepagency)
 *      network mb and MS determination from reported amplitudes
 *      RSTT Pg/Lg, Pn/Sn travel-time predictions
 *
 *  Reference
 *      Bondar, I. and D. Storchak, (2011).
 *      Improved location procedures at the International Seismological Centre,
 *      Geophys. J. Int., 186, 1220-1244,
 *      DOI: 10.1111/j.1365-246X.2011.05107.x
 *
 *  Environment variables
 *     QETC - directory pathname for data files. If not exists, defaults
 *            to current working directory.
 *     QEDIT - directory pathname for the instruction file (ISC specific).
 *            If not exists, reads instructions from stdin.
 *     QREVISE - directory pathname for the log file (ISC specific).
 *     PGHOSTADDR, PGPORT, PGDATABASE, PGUSER, PGPASSWORD - Postgres specific.
 *     SLBMROOT - RSTT software package root directory.
 *
 *  Input files
 *     Instructions from file or stdin
 *         If a command line argument is given then it should be yyyymmv and
 *         the instruction file name will be formed out of that and environment
 *         variable $QEDIT, otherwise will read instructions from stdin.
 *     Configuration parameters
 *         $QETC/iscloc/config.txt
 *     Velocity model and phase specific parameters (ttime_table = ak135 or jb)
 *         $QETC/iscloc/<ttime_table>_model.txt
 *     Travel-time tables
 *         $QETC/<ttime_table>/[*].tab
 *     Ellipticity correction coefficients for ak135
 *         $QETC/ak135/ELCOR.dat
 *     Topography file for bounce point corrections
 *         $QETC/topo/etopo5_bed_g_i2.bin (ETOPO1 resampled to 5'x5' resolution)
 *     Flinn-Engdahl regionalization (1995)
 *         $QETC/FlinnEngdahl/FE.dat
 *     Default depth grid
 *         $QETC/FlinnEngdahl/default.depth0.5.grid
 *     Default depth for Flinn-Engdahl regions
 *         $QETC/FlinnEngdahl/grn_default_depth.<ttime_table>.dat
 *         For locations where no default depth grid point exists.
 *     Generic variogram model for data covariance matrix
 *         $QETC/variogram/variogram.model
 *     Magnitude attenuation Q(d,h) curves for mb calculation
 *         $QETC/magnitude/GRmbQ.dat (Gutenberg-Richter, 1956) or
 *         $QETC/magnitude/VCmbQ.dat (Veith-Clawson, 1972)
 *         $QETC/magnitude/MBmbQ.dat (Murphy-Barker, 2003)
 *
 *  Configuration parameters
 *     Output files
 *         logfile = stdout - default - overwritten by yyyymmv if given
 *         errfile = stderr - for errors
 *     Database options (ISC specific)
 *         update_db = 1    - write results to database?
 *         nextid_db = isc  - get new ids from this account
 *         repid = 100      - 'reporter' field for new hypocentres and assocs
 *         out_agency = ISC - author for new hypocentres and assocs
 *         in_agency = ISC  - author for input assocs
 *     ISF input/output
 *         isf_stafile      - pathname of station coordinates file
 *     Travel time table [ak135|jb]
 *         ttime_table = ak135 - travel time table name
 *     ETOPO parameters
 *         etopofile = etopo5_bed_g_i2.bin - ETOPO file name
 *         etoponlon = 4321                - ETOPO longitude samples
 *         etoponlat = 2161                - ETOPO latitude samples
 *         etopores = 0.0833333            - ETOPO resolution
 *     Depth resolution
 *         mindepthpha = 5  - min number of depth phases for depth-phase depth
 *         ndepagency = 2   - min number of depth-phase reporting agencies
 *         localdist = 0.2  - max local distance to be considered [deg]
 *         minlocalsta = 1  - min number of local defining stations
 *         spdist = 3.      - max distance for defining S-P phase pairs to be
 *                            considered [deg]
 *         min_s_p = 5      - min number of defining S-P phase pairs
 *         min_corepha = 5  - min number of defining core reflection phases
 *         maxdeperror_shallow = 30. - max depth error for crustal free-depth
 *         maxdeperror_deep = 60.    - max depth error for deep free-depth
 *         default_depth = 0   - used if seed hypocentre depth is NULL
 *     Iteration control
 *         min_iter = 4             - min number of iterations
 *         max_iter = 20            - max number of iterations
 *         min_phases = 4           - min number of phases
 *         sigmathres = 6.          - used to exclude phases from solution
 *         do_correlated_errors = 1 - account for correlated errors?
 *         allow_damping = 1        - allow damping in LSQR iterations?
 *         confidence = 90.         - confidence level for uncertainties
 *     Agencies whose hypocenters not to be used in setting the initial guess
 *         nohypo_agencies = UNK,NIED,HFS,HFS1,HFS2,NAO,LAO
 *                           # UNK   - unknown agency
 *                           # NIED  - truncates origin time to the minute
 *                           # HFS, NAO, LAO - single array locations
 *     Neighbourhood Algorithm parameters
 *         do_gridsearch = 0  - perform NA grid search?
 *         na_radius = 5.     - search radius [deg] around initial epicentre
 *         na_deptol = 300.   - search radius [km] around initial depth
 *         na_ottol = 30.     - search radius [s] around initial origin time
 *         na_lpnorm = 1.     - p-value for Lp-norm to compute misfit
 *         na_itermax = 5     - max number of iterations
 *         na_nsamplei = 700  - size of initial sample
 *         na_nsample = 100   - size of subsequent samples
 *         na_ncells = 25     - number of cells to be resampled
 *         iseed = 5590       - random number seed
 *     Magnitude calculations
 *         mbQ_table = GR           - magnitude correction table [GR,VC,MB,none]
 *         body_mag_min_dist = 21.  - min delta for calculating mb
 *         body_mag_max_dist = 100. - max delta for calculating mb
 *         surf_mag_min_dist = 20.  - min delta for calculating Ms
 *         surf_mag_max_dist = 160. - max delta for calculating Ms
 *         body_mag_min_per  = 0.   - min period for calculating mb
 *         body_mag_max_per  = 3.   - max period for calculating mb
 *         surf_mag_min_per  = 10.  - min period for calculating Ms
 *         surf_mag_max_per  = 60.  - max period for calculating Ms
 *         surf_mag_max_depth = 60. - max depth for calculating Ms
 *         pertol            = 5.   - MSH period tolerance around MSZ period
 *         mag_range_warn_thresh = 2.2  - allowable range around network mag
 *     RSTT-specific parameters
 *         rstt_model         - pathname of RSTT model file
 *         use_RSTT_PgLg = 0  - use RSTT Pg/Lg predictions?
 *
 *  Instructions
 *     isloc locates events with evids given one per line
 *     in the instruction file.
 *         verbose           - verbosity level (0..5)
 *         depth             - an agency code or a number
 *         lat,lon           - an agency code or a number
 *         time              - an agency code or a number
 *         fix_depth         - an agency code or a number
 *         fix_location      - an agency code
 *         fix_time          - an agency code or a number
 *         fix_hypo          - an agency code
 *         fix_depth_depdp   - presence sets flag to 1
 *         fix_depth_default - presence sets flag to 1
 *         indb              - input DB account [default=PGUSER]
 *         outdb             - output DB account [default=PGUSER]
 *         isf_infile        - read ISF input from here
 *         isf_outfile       - write ISF output to here
 *         write_gridsearch_results - write NA results to file
 *     The options below can override external parameters in the config file:
 *         mindepthpha       - min # of depth phases for depth-phase depth
 *         ndepagency        - min # of agencies reporting depth phases
 *         localdist         - max local distance (degs)
 *         minlocalsta       - min number of stations within localdist
 *         spdist            - max S-P distance (degs)
 *         min_s_p           - min number of S-P phase pairs
 *         min_corepha       - min number of core reflections ([PS]c[PS]
 *         iseed             - random number initial seed
 *         update_db         - update DB?
 *         do_gridsearch     - perform initial grid search?
 *         na_radius         - search radius [deg] around initial epicentre
 *         na_deptol         - search radius [km] around initial depth
 *         na_ottol          - search radius [s] around initial origin time
 *         na_nsamplei       - size of initial sample
 *         na_nsample        - size of subsequent samples
 *         na_ncells         - number of cells to be resampled
 *         na_itermax        - max number of iterations
 *         do_correlated_errors - account for correlated error structure?
 *         isf_stafile       - pathname of station coordinates file
 *         pertol            - MSH period tolerance around MSZ period
 *         rstt_model        - pathname of RSTT model file
 *         use_RSTT_PgLg     - use RSTT Pg/Lg predictions?
 *
 *  For each event try 2 options:
 *     0 - free depth (if there is depth resolution)
 *     1 - fix depth to region-dependent default depth (if option 0 fails)
 *  Further options can be requested via instruction arguments:
 *     2 - fix depth to a reported hypocentre's depth
 *     3 - fix depth to depth-phase depth if possible
 *     4 - fix epicentre
 *     5 - fix depth and epicentre
 *     6 - fix hypocentre (in this case simply calculate the residuals)
 *  If convergence is reached, no further options are tried.
 *
 *  ISC terminology
 *     prime
 *        the preferred hypocentre among the various reported hypocentres
 *     reading
 *        the entire set of phases/amplitudes reported by an agency
 *        at a particular station for a particular event
 *        submitted in a single report
 *        (e.g. Pg, Pn, Sn at ARCES reported by the IDC for an event)
 *     phase
 *        arrival/amplitude data reported for a phase pick
 */

#include "rstt.h"

/*
 *
 * Global variables for entries from the model file
 *
 */
double moho;                                               /* depth of Moho */
double conrad;                                           /* depth of Conrad */
double max_depth_km;                                /* max hypocenter depth */
double psurfvel;                   /* Pg velocity for elevation corrections */
double ssurfvel;                   /* Sg velocity for elevation corrections */
char no_resid_phase[MAXNUMPHA][PHALEN];    /* no residuals for these phases */
int no_resid_phase_num;                     /* number of no-residual phases */
char mb_phase[MAXNUMPHA][PHALEN];        /* phases used in mb determination */
int mb_phase_num;                                    /* number of mb phases */
char ms_phase[MAXNUMPHA][PHALEN];        /* phases used in Ms determination */
int ms_phase_num;                                    /* number of Ms phases */
PHASEMAP phase_map[MAXPHACODES];        /* reported to IASPEI phase mapping */
int phase_map_num;                         /* number of phases in phase_map */
PHASEWEIGHT phase_weight[MAXNUMPHA];          /* prior measerror for phases */
int phase_weight_num;                   /* number of phases in phase_weight */
char allowable_phase[MAXTTPHA][PHALEN];                 /* allowable phases */
int no_allowable_phase_num;                   /* number of allowable phases */
char firstPphase[MAXTTPHA][PHALEN];              /* first-arriving P phases */
int firstPphase_num;                   /* number of first-arriving P phases */
char firstSphase[MAXTTPHA][PHALEN];              /* first-arriving S phases */
int firstSphase_num;                   /* number of first-arriving S phases */
char firstPopt[MAXTTPHA][PHALEN];                /* optional first P phases */
int firstPopt_num;                     /* number of optional first P phases */
char firstSopt[MAXTTPHA][PHALEN];                /* optional first S phases */
int firstSopt_num;                     /* number of optional first S phases */

/*
 *
 * Global variables for entries from the configuration file
 *
 *
 * I/O
 */
char logfile[FILENAMELEN];                                  /* log filename */
char errfile[FILENAMELEN];                                /* error filename */
char isf_stafile[FILENAMELEN];     /* station file when not reading from db */
char isf_outfile[FILENAMELEN];                       /* output ISF filename */
char isf_infile[FILENAMELEN];                         /* input ISF filename */
int update_db;                                  /* write result to database */
char nextid_db[24];              /* get new unique ids from this DB account */
int repid;              /* reporter id field for new hypocentres and assocs */
char out_agency[VALLEN];           /* author for new hypocentres and assocs */
char in_agency[VALLEN];                          /* author for input assocs */
/*
 * TT
 */
char ttime_table[VALLEN];                         /* travel time table name */
double default_depth;              /* used if seed hypocentre depth is NULL */
/*
 * iteration control
 */
int min_iter;                                   /* min number of iterations */
int max_iter;                                   /* max number of iterations */
int min_phases;                                     /* min number of phases */
double sigmathres;                       /* to exclude phases from solution */
int do_correlated_errors;                  /* account for correlated errors */
int allow_damping;                      /* allow damping in LSQR iterations */
double confidence;                    /* confidence level for uncertainties */
/*
 *  depth-phase depth solution requirements
 */
int mindepthpha;        /* min number of depth phases for depth-phase depth */
int ndepagency;            /* min number of agencies reporting depth phases */
/*
 *  depth resolution
 */
double localdist;                              /* max local distance (degs) */
int minlocalsta;                 /* min number of stations within localdist */
double spdist;                                   /* max S-P distance (degs) */
int min_s_p;                               /* min number of S-P phase pairs */
int min_corepha;               /* min number of core reflections ([PS]c[PS] */
/*
 * maximum allowable depth error for free depth solutions
 */
double maxdeperror_shallow;
double maxdeperror_deep;
/*
 * magnitudes
 */
char mbQ_table[VALLEN];                       /* magnitude correction table */
double body_mag_min_dist;                   /* min delta for calculating mb */
double body_mag_max_dist;                   /* max delta for calculating mb */
double surf_mag_min_dist;                   /* min delta for calculating Ms */
double surf_mag_max_dist;                   /* max delta for calculating Ms */
double body_mag_min_per;                   /* min period for calculating mb */
double body_mag_max_per;                   /* max period for calculating mb */
double surf_mag_min_per;                   /* min period for calculating Ms */
double surf_mag_max_per;                   /* max period for calculating Ms */
double surf_mag_max_depth;                  /* max depth for calculating Ms */
double mag_range_warn_thresh;           /* warn about outliers beyond range */
double pertol;                    /* MSH period tolerance around MSZ period */
/*
 * NA search parameters
 */
int do_gridsearch;              /* perform grid search for initial location */
int write_gridsearch_results;                      /* write results to file */
double na_radius;         /* search radius (degrees) around prime epicenter */
double na_deptol;                  /* search radius (km) around prime depth */
double na_ottol;              /* search radius (s) around prime origin time */
double na_lpnorm;                     /* p-value for norm to compute misfit */
int na_itermax;                                 /* max number of iterations */
int na_nsamplei;                                  /* size of initial sample */
int na_nsample;                               /* size of subsequent samples */
int na_ncells;                           /* number of cells to be resampled */
long iseed;                                           /* random number seed */
/*
 * agencies whose hypocenters not to be used in setting initial hypocentre
 */
char nohypoagency[MAXBUF][AGLEN];
int numnohypoagency;
/*
 * ETOPO parameters
 */
char etopofile[FILENAMELEN];                     /* filename for ETOPO file */
int etoponlon;                      /* number of longitude samples in ETOPO */
int etoponlat;                       /* number of latitude samples in ETOPO */
double etopores;                                       /* cellsize in ETOPO */
/*
 * RSTT parameters
 */
char rstt_model[FILENAMELEN];                                 /* RSTT model */
int use_RSTT_PgLg;                            /* use RSTT Pg/Lg predictions */
/*
 *
 * Global variables for entries from the instruction file
 *
 */
char indb[24];                    /* read data from this DB account, if any */
char outdb[24];                 /* write results to this DB account, if any */
int verbose;                                             /* verbosity level */
/*
 *
 * Global variables
 *
 */
FILE *logfp = (FILE *)NULL;                     /* file pointer to log file */
FILE *errfp = (FILE *)NULL;                   /* file pointer to error file */
#ifdef WITH_DB
PGconn *conn;                                              /* DB connection */
#endif
struct timeval t0;
/*
 * station list from ISF file
 */
STAREC *stationlist;
int numsta;                            /* number of stations in stationlist */
/*
 * agencies contributing phase data
 */
char agencies[MAXBUF][AGLEN];
int numagencies;
/*
 * list of phases with ak135 TT prediction
 */
int phaseTT_num = MAXTTPHA;
char phaseTT[MAXTTPHA][PHALEN] = {
    "firstP", "firstS",
    "P", "Pn", "Pg", "Pb",
    "PKPdf", "PKPab", "PKPbc", "PKSdf", "PKSab", "PKSbc",
    "S", "Sn", "Sg", "Sb",
    "SKPdf", "SKPab", "SKPbc", "SKSdf", "SKSac",
    "pP", "pwP", "sP", "pPKPdf", "PcP", "PcS", "ScP", "ScS",
    "Pdif", "PKKPdf", "PKKPab", "PKKPbc", "PKKSdf", "PKKSab", "PKKSbc",
    "Sdif", "SKKPdf", "SKKPab", "SKKPbc", "SKKSdf", "SKKSac", "PKPdif",
    "PKiKP", "PP", "PbPb", "PnPn", "PgS", "PnS", "P'P'df", "P'P'ab", "P'P'bc",
    "SKiKP", "SS", "SbSb", "SnSn", "SPg", "SPn", "S'S'df", "S'S'ac",
    "pS", "sS", "pPb", "sPb", "sSb", "pPn", "sPn", "sSn",
    "pPdif", "sPdif", "pSdif", "sSdif", "pPKiKP", "sPKiKP",
    "pPKPab", "sPKPab", "pPKPbc", "sPKPbc", "sPKPdf",
    "pSKSac", "sSKSac", "pSKSdf", "sSKSdf", "pPKPdif", "sPKPdif",
    "PPP", "PSP", "PSS", "PS", "SPP", "SSS", "SP", "SSP", "PgPg", "SgSg"
};
/*
 * Error codes
 */
int errorcode;
char *errorcodes[] = {
    "unknown error, please consult log file",
    "memory allocation error",
    "could not open file",
    "bad instruction",
    "diverging solution",
    "insufficient number of phases",
    "insufficient number of independent phases",
    "phase loss",
    "slow convergence",
    "singular G matrix",
    "abnormally ill-conditioned problem",
    "invalid station code",
};

/*
 *
 * Local functions
 *
 */
static int open_log_err_files(char *logfile, char *errfile, char *yyyymm,
                              char version);
/*
 *
 * main body
 *
 */
int main(int argc, char *argv[])
{
#ifdef WITH_DB
    PGresult *res_set = (PGresult *)NULL;
#endif
    char sql[1024];
    FILE *instructfp;
    FILE *isfin, *isfout;
    char *buffer;
    char configdir[FILENAMELEN];
    char editdir[FILENAMELEN];
    char configfile[FILENAMELEN];
    char instructfile[FILENAMELEN];
    char instruction[LINLEN];
    char system_command[LINLEN];
    char usage[LINLEN];
    char yyyymm[7];                                    /* From command line */
    char version;                                      /* From command line */

    EVREC e;                                                /* event record */
    SOLREC s;                                            /* solution record */
    HYPREC *h = (HYPREC *)NULL;                       /* hypocenter records */
    PHAREC *p = (PHAREC *)NULL;                            /* phase records */
    TT_TABLE *tt_tables = (TT_TABLE *)NULL;     /* ak135 travel-time tables */
    EC_COEF *ec = (EC_COEF *)NULL;   /* ellipticity correction coefficients */
    VARIOGRAM variogram;                         /* generic variogram model */
    MAGQ mbQ;                                /* magnitude attenuation table */
    STAMAG *stamag_mb = (STAMAG *)NULL;     /* mb station magnitude records */
    STAMAG *stamag_ms = (STAMAG *)NULL;     /* MS station magnitude records */
    RDMAG *rdmag_mb = (RDMAG *)NULL;        /* mb reading magnitude records */
    RDMAG *rdmag_ms = (RDMAG *)NULL;        /* MS reading magnitude records */
    MSZH *mszh = (MSZH *)NULL;  /* MS vertical|horizontal magnitude records */
    FE fe;                       /* Flinn-Engdahl geographic region numbers */
    double **depthgrid = (double **)NULL;             /* default depth grid */
    double *grn_depth = (double *)NULL;            /* default depths by grn */
    short int **topo = (short int **)NULL;    /* ETOPO bathymetry/elevation */

    int total = 0, fail = 0, opt[7];                /* counters for results */
    int ismbQ = 0;                          /* apply magnitude attenuation? */
    int isf = 0;                                    /* ISF text file input? */
    int database = 0;
    int ngrid = 0, i;
    int num_ecphases = 0;
    double gres = 1., d = DEG_TO_RAD * MAX_RSTT_DIST;
    struct timeval t00;
    int mindepthpha_cf = 5, ndepagency_cf = 2, minlocalsta_cf = 1;
    int min_s_p_cf = 5, min_corepha_cf = 5;
    double localdist_cf = 0.2, spdist_cf = 3., pertol_cf = 5.;
    int do_gridsearch_cf = 0, do_correlated_errors_cf = 1;
    int update_db_cf = 0;
    long iseed_cf = 5590;
    double na_radius_cf = 5., na_deptol_cf = 300., na_ottol_cf = 30.;
    int na_nsamplei_cf = 700, na_nsample_cf = 100, na_ncells_cf = 25;
    int na_itermax_cf = 5;
    int use_RSTT_PgLg_cf = 0;
    int depfix_cf, surfix_cf, hypofix_cf;
    char rstt_model_cf[FILENAMELEN];
    char magbloc[10*LINLEN];
/*
 *  set timezone to get epoch times right and other inits
 */
    setenv("TZ", "", 1);
    tzset();
    strcpy(isf_outfile, "");
    for (i = 0; i < 8; i++) opt[i] = 0;
    gettimeofday(&t00, NULL);
    gettimeofday(&t0, NULL);
    verbose = 0;
    errorcode = 0;
/*
 *  Get command line argument and open instruction file if any
 */
    strcpy(usage, "usage: iscloc [isf|isf2|yyyymmv|-] [instructions]\n");
    if (argc == 1) {
/*
 *      no argument is given, abort!
 */
        fprintf(stderr, "ABORT: No argument is given!\n%s\n", usage);
        fprintf(stderr, "EVENT %.6f\n\n", secs(&t0));
        exit(1);
    }
    else if (strcmp(argv[1], "-") == 0) {
/*
 *      instructions will be taken from stdin
 *      log will be written to the file specified in the config.txt file.
 */
        yyyymm[0] = '\0';
        strcpy(instructfile, "stdin");
        instructfp = stdin;
    }
    else if (streq(argv[1], "isf")) {
/*
 *      data will be read from an ISF input file
 *      instructions will be taken from stdin
 *      log will be written to the file specified in the config.txt file.
 */
        yyyymm[0] = '\0';
        strcpy(instructfile, "stdin");
        instructfp = stdin;
        isf = 1;
    }
    else if (streq(argv[1], "isf2")) {
/*
 *      data will be read from an ISF2 input file
 *      instructions will be taken from stdin
 *      log will be written to the file specified in the config.txt file.
 */
        yyyymm[0] = '\0';
        strcpy(instructfile, "stdin");
        instructfp = stdin;
        isf = 2;
    }
    else {
/*
 *      Use command line argument to create the filenames for
 *          the instruction file and the log file.
 *      Instructions given in the command line are ignored.
 *      This is the ISC operational mode.
 *
 *      editdir - the directory to find the instruction file is
 *          based on $QEDIT. If the $QEDIT environment variable doesn't exist,
 *          editdir defaults to the current working directory.
 */
        for (i = 0; i < 6; i++) {
            if (!isdigit(argv[1][i])) {
                fprintf(stderr, "%s", usage);
                exit(1);
            }
            yyyymm[i] = argv[1][i];
        }
        yyyymm[6] = '\0';
        if (!islower(argv[1][6])) {
            fprintf(stderr, "%s", usage);
            fprintf(stderr, "EVENT %.6f\n\n", secs(&t0));
            exit(1);
        }
        version = argv[1][6];
        if ((buffer = getenv("QEDIT")) == NULL) {
            fprintf(stderr, "QEDIT not defined\n");
            fprintf(stderr, "EVENT %.6f\n\n", secs(&t0));
            exit(1);
        }
        strcpy(editdir, buffer);
        sprintf(instructfile, "%s/%s/%s%c.in", editdir, yyyymm, yyyymm,
                version);
        if ((instructfp = fopen(instructfile, "r")) == NULL) {
            fprintf(stderr, "Cannot open instruction file %s\n", instructfile);
            fprintf(stderr, "EVENT %.6f\n\n", secs(&t0));
            exit(1);
        }
    }
/*
 *
 *  Get configuration directory name
 *      If the $QETC environment variable doesn't exist,
 *      configdir defaults to the current working directory.
 *
 */
    if ((buffer = getenv("QETC"))) sprintf(configdir, "%s", buffer);
    else                           strcpy(configdir, ".");
/*
 *
 *  Read configuration file from configdir/iscloc directory
 *
 */
    sprintf(configfile, "%s/iscloc/config.txt", configdir);
    printf("read_config: %s\n", configfile);
    if (read_config(configfile)) {
        fprintf(stderr, "EVENT %.6f\n\n", secs(&t0));
        exit(1);
    }
/*
 *  save config pars that could be overriden by an instruction
 */
    mindepthpha_cf = mindepthpha;
    ndepagency_cf = ndepagency;
    localdist_cf = localdist;
    minlocalsta_cf = minlocalsta;
    spdist_cf = spdist;
    min_s_p_cf = min_s_p;
    min_corepha_cf = min_corepha;
    do_gridsearch_cf = do_gridsearch;
    na_radius_cf = na_radius;
    na_deptol_cf = na_deptol;
    na_ottol_cf = na_ottol;
    na_nsamplei_cf = na_nsamplei;
    na_nsample_cf = na_nsample;
    na_ncells_cf = na_ncells;
    na_itermax_cf = na_itermax;
    update_db_cf = update_db;
    iseed_cf = iseed;
    pertol_cf = pertol;
    do_correlated_errors_cf = do_correlated_errors;
    use_RSTT_PgLg_cf = use_RSTT_PgLg;
    strcpy(rstt_model_cf, rstt_model);
/*
 *
 *  Open error and log files
 *
 */
    if (open_log_err_files(logfile, errfile, yyyymm, version)) {
        fprintf(stderr, "EVENT %.6f\n\n", secs(&t0));
        exit(1);
    }
/*
 *
 *  ISF input file
 *      only the first line of the instruction file is interpreted
 *      can run without instruction file
 *
 */
    if (isf) {
        e.evid = NULLVAL;
        e.depth_agency[0] = '\0';
        e.location_agency[0] = '\0';
        e.time_agency[0]     = '\0';
        e.hypo_agency[0]     = '\0';
        e.start_depth = e.start_time = NULLVAL;
        e.start_lat = e.start_lon = NULLVAL;
        e.surface_fix = e.hypo_fix = 0;
        e.time_fix = e.depth_fix = 0;
        e.depth_fix_editor = e.epi_fix = 0;
        e.fix_depth_default = e.fix_depth_depdp = 0;
        depfix_cf = surfix_cf = hypofix_cf = 0;
        if (fgets(system_command, LINLEN, instructfp)) {
            dropspace(system_command, instruction);
            fprintf(logfp, "Instruction: %s", instruction);
            fprintf(errfp, "instruction: %s", instruction);
            if (strlen(instruction) == LINLEN - 1) {
                fprintf(errfp, "Instruction line too long: %s", instruction);
                fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
                exit(1);
            }
            if (read_instruction(instruction, &e, isf)) {
                fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
                exit(1);
            }
            depfix_cf = e.depth_fix;
            surfix_cf = e.surface_fix;
            hypofix_cf = e.hypo_fix;
        }
        ndepagency = 1;          /* there is no agency info in an ISF file! */
/*
 *
 *      Open ISF input/output files if any
 *
 */
        if (isf_infile[0]) {
            if (streq(isf_infile, "stdin"))
                isfin = stdin;
            else if ((isfin = fopen(isf_infile, "r")) == NULL) {
                fprintf(errfp, "Cannot open isf input file %s\n", isf_infile);
                fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
                exit(1);
            }
        }
        if (isf_outfile[0]) {
            if (streq(isf_outfile, "stdout"))
                isfout = stdout;
            else if ((isfout = fopen(isf_outfile, "w")) == NULL) {
                fprintf(errfp, "Cannot open isf output file %s\n", isf_outfile);
                fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
                exit(1);
            }
        }
/*
 *      read ISF station information from file
 */
        if (verbose) fprintf(logfp, "    read_stafile\n");
        if (isf == 2) {
            if (read_stafile_isf2(&stationlist)) {
                fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
                exit(1);
            }
        }
        else {
            if (read_stafile(&stationlist)) {
                fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
                exit(1);
            }
        }
    }
/*
 *
 *  Read various data files from config directory
 *
 */
    if (verbose) fprintf(logfp, "read_data_files (%.4f)\n", secs(&t0));
    if (read_data_files(configdir, &ismbQ, &mbQ, &fe, &grn_depth, &gres, &ngrid,
                        &depthgrid, &topo, &num_ecphases, &ec, &tt_tables,
                        &variogram)) {
        fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
        if (isf) Free(stationlist);
        exit(1);
    }
    if (verbose) fprintf(logfp, "read_data_files (%.4f) done\n", secs(&t0));
#ifdef WITH_RSTT
/*
 *
 *  Initialize SLBM and read RSTT model from
 *
 */
    slbm_shell_create();
    if (slbm_shell_loadVelocityModelBinary(rstt_model)) {
        fprintf(errfp, "ABORT: Cannot open RSTT model %s\n", rstt_model);
        fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
        slbm_shell_delete();
        if (isf) Free(stationlist);
        goto abort;
    }
    slbm_shell_setMaxDistance(&d);
    strcpy(buffer, "NATUTAL_NEIGHBOR");
    slbm_shell_setInterpolatorType(buffer);
#endif
/*
 *
 *  Read data from ISF input file
 *
 */
    if (isf) {
/*
 *      Event loop: Read next event from ISF file
 */
        while (!read_isf(isfin, isf, &e, &h, &p, stationlist, &stamag_mb,
                         &stamag_ms, &rdmag_mb, &rdmag_ms, &mszh, magbloc)) {
            if (verbose) fprintf(logfp, "    eventloc (%.4f)\n", secs(&t0));
/*
 *          locate event
 */
            if (eventloc(isf, database, &total, &fail, opt, &e, h, &s, p,
                         stamag_mb, stamag_ms, rdmag_mb, rdmag_ms, mszh,
                         ismbQ, &mbQ, ec, tt_tables, &variogram, gres, ngrid,
                         depthgrid, &fe, grn_depth, topo, isfout, magbloc)) {
                fprintf(logfp, "CAUTION: No solution found due to %s\n",
                        errorcodes[errorcode]);
                fprintf(errfp, "CAUTION: No solution found due to %s\n",
                        errorcodes[errorcode]);
            }
            fprintf(logfp, "EVENT %.6f %d %d\n", secs(&t0), e.evid, s.numphas);
            Free(h); Free(p);
            Free(stamag_mb); Free(stamag_ms);
            Free(rdmag_mb); Free(rdmag_ms); Free(mszh);
            fprintf(logfp, "\n\n");
            if (e.surface_fix)
                e.start_depth = e.start_time = NULLVAL;
            e.depth_fix = depfix_cf;
            e.surface_fix = surfix_cf;
            e.hypo_fix = hypofix_cf;
        }
        fclose(isfin);
        if (isf_outfile[0])
            fclose(isfout);
        Free(stationlist);
    }
/*
 *
 *  Read data from database
 *
 */
#ifdef WITH_DB
    else {
/*
 *      Establish and test DB connection
 */
        if (verbose)
            fprintf(logfp, "    establish DB connection\n");
        if (pgsql_conn()) {
            fprintf(logfp, "ABORT: cannot establish DB connection!\n");
            fprintf(errfp, "ABORT: cannot establish DB connection!\n");
            fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
            goto abort;
        }
        sprintf(sql, "SELECT NOW()");
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("main: error: ");
        else if (PQntuples(res_set) == 1) {
            fprintf(logfp, "    %s\n", PQgetvalue(res_set, 0, 0));
        }
        PQclear(res_set);
        if (conn == NULL) {
            fprintf(logfp, "ABORT: cannot establish DB connection!\n");
            fprintf(errfp, "ABORT: cannot establish DB connection!\n");
            fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
            goto abort;
        }
        database = 1;
/*
 *      Event loop: get next event instruction line
 */
        e.evid = 0;
        while (fgets(system_command, LINLEN, instructfp)) {
            errorcode = 0;
            dropspace(system_command, instruction);
            fprintf(logfp, "Instruction: %s", instruction);
            fprintf(errfp, "instruction: %s", instruction);
            if (strlen(instruction) == LINLEN - 1) {
                fprintf(errfp, "Instruction line too long: %s", instruction);
                fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
                goto abort;
            }
/*
 *          restore config pars that could be overriden by an instruction
 */
            mindepthpha = mindepthpha_cf;
            ndepagency = ndepagency_cf;
            localdist = localdist_cf;
            minlocalsta = minlocalsta_cf;
            spdist = spdist_cf;
            min_s_p = min_s_p_cf;
            min_corepha = min_corepha_cf;
            do_gridsearch = do_gridsearch_cf;
            na_radius = na_radius_cf;
            na_deptol = na_deptol_cf;
            na_ottol = na_ottol_cf;
            na_nsamplei = na_nsamplei_cf;
            na_nsample = na_nsample_cf;
            na_ncells = na_ncells_cf;
            na_itermax = na_itermax_cf;
            update_db = update_db_cf;
            iseed = iseed_cf;
            pertol = pertol_cf;
            do_correlated_errors = do_correlated_errors_cf;
            verbose = 0;
            strcpy(indb, "");
            strcpy(outdb, "");
            strcpy(isf_outfile, "");
            write_gridsearch_results = 0;
            use_RSTT_PgLg = use_RSTT_PgLg_cf;
            strcpy(rstt_model, rstt_model_cf);
/*
 *          parse instruction
 */
            e.evid = NULLVAL;
            if (read_instruction(instruction, &e, isf)) {
                fprintf(logfp, "WARNING: bad instruction line!\n");
                fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
                continue;
            }
/*
 *          Open ISF output file if requested
 */
            if (isf_outfile[0]) {
                if (streq(isf_outfile, "stdout"))
                    isfout = stdout;
                else if ((isfout = fopen(isf_outfile, "w")) == NULL) {
                    fprintf(errfp, "Cannot open isf output file %s\n",
                            isf_outfile);
                    fprintf(logfp, "EVENT %.6f\n\n", secs(&t0));
                    continue;
                }
            }
/*
 *          Read hypocenters, phases, and associations from database
 */
            if (verbose) fprintf(logfp, "get_data (%.4f)\n", secs(&t0));
            if (get_data(&e, &h, &p, &stamag_mb, &stamag_ms,
                         &rdmag_mb, &rdmag_ms, &mszh)) {
                fprintf(logfp, "EVENT %.6f %d\n", secs(&t0), e.evid);
                if (isf_outfile[0])
                    fclose(isfout);
                continue;
            }
            if (verbose) fprintf(logfp, "get_data (%.4f) done\n", secs(&t0));
/*
 *          locate event
 */
            if (eventloc(isf, database, &total, &fail, opt, &e, h, &s, p,
                         stamag_mb, stamag_ms, rdmag_mb, rdmag_ms, mszh,
                         ismbQ, &mbQ, ec, tt_tables, &variogram, gres, ngrid,
                         depthgrid, &fe, grn_depth, topo, isfout, magbloc)) {
                fprintf(logfp, "CAUTION: No solution found due to %s\n",
                        errorcodes[errorcode]);
                fprintf(errfp, "CAUTION: No solution found due to %s\n",
                        errorcodes[errorcode]);
            }
            fprintf(logfp, "EVENT %.6f %d %d\n", secs(&t0), e.evid, s.numphas);
            Free(h); Free(p);
            Free(stamag_mb); Free(stamag_ms);
            Free(rdmag_mb); Free(rdmag_ms); Free(mszh);
            if (isf_outfile[0])
                fclose(isfout);
            fprintf(logfp, "\n\n");
        }
    }
#endif
/*
 *  End of event loop
 */
abort:
    if (strcmp(instructfile, "stdin"))
        fclose(instructfp);
/*
 *  disconnect from database
 */
#ifdef WITH_DB
    if (database)
        pgsql_disconn();
#endif
/*
 *  free default depth grid and etopo
 */
    free_fe(&fe);
    free_matrix(depthgrid);
    free_i2matrix(topo);
    Free(grn_depth);
/*
 *  free travel-time tables
 */
    if (tt_tables != NULL)
        free_tt_tbl(tt_tables);
/*
 *  free ellipticity correction coefficients
 */
    if (ec != NULL)
        free_elcor_tbl(ec, num_ecphases);
/*
 *  free mbQ
 */
    if (ismbQ) {
        Free(mbQ.deltas);
        Free(mbQ.depths);
        free_matrix(mbQ.q);
    }
/*
 *  free variogram
 */
    free_variogram(&variogram);
/*
 *  delete SLBM instance
 */
#ifdef WITH_RSTT
    slbm_shell_delete();
#endif
/*
 *  report on totals
 */
    fprintf(logfp, "\nTotals: option 0: %d 1: %d 2: %d 3: %d 4: %d 5: %d 6: %d",
            opt[0], opt[1], opt[2], opt[3], opt[4], opt[5], opt[6]);
    fprintf(logfp, " converged: %d failed: %d time %.2f\n",
            total, fail, secs(&t00));
/*
 *  Close output files, if any.
 */
    if (strcmp(logfile, "stderr") && strcmp(logfile, "stdout"))
        fclose(logfp);
    if (strcmp(errfile, "stderr") && strcmp(errfile, "stdout"))
        fclose(errfp);
    exit(0);
}

/*
 *  Title:
 *     open_log_err_files
 *  Synopsis:
 *     Open files for log and error messages
 *  Input Arguments:
 *     logfile - pathname for logfile
 *     errfile - pathname for error log
 *     yyyymm, version - if non-null log filename will be constructed from
 *               these variables in $QREVISE directory (ISC-specific)
 *  Return:
 *     0/1 for success/failure.
 */
static int open_log_err_files(char *logfile, char *errfile, char *yyyymm,
                              char version)
{
    int i;
    char revisedir[FILENAMELEN];
    char *buffer;
/*
 *  Open file for error messages
 */
    if (yyyymm[0] != '\0') {
        /* Get QREVISE environment variable. */
        if ((buffer = getenv("QREVISE")) == NULL) {
            fprintf(errfp, "QREVISE not defined\n");
            return 1;
        }
        strcpy(revisedir, buffer);
        sprintf(errfile, "%s/%s%c.iscloc.err", revisedir, yyyymm, version);
        if ((errfp = fopen(errfile, "w")) == NULL) {
            fprintf(stderr, "Can't open error file %s\n", errfile);
            return 1;
        }
    }
    else if (errfile[0]) {
        if (streq(errfile, "stdout"))
            errfp = stdout;
        else if (streq(errfile, "stderr"))
            errfp = stderr;
        else {
            /* First > is meaningless - always write to errfile. */
            if (errfile[0] == '>') {
                i = 0;
                while ((errfile[i] = errfile[i+1])) i++;
            }
            /* But if there are two then want append. */
            if (errfile[0] == '>') {
                i = 0;
                while ((errfile[i] = errfile[i+1])) i++;
                if ((errfp = fopen(errfile, "a")) == NULL) {
                    fprintf(stderr, "ERROR: Can't open error file\n");
                    return 1;
                }
            }
            else {
                if ((errfp = fopen(errfile, "w")) == NULL) {
                    fprintf(stderr, "ERROR: Can't open error file\n");
                    return 1;
                }
            }
        }
    }
    else {
        fprintf(stderr, "ERROR: no error file given\n");
        return 1;
    }
/*
 *  Open logfile
 *      If command line argument given then use that to form filename.
 *      Default to location in config file - could be stdout/stderr.
 */
    if (yyyymm[0] != '\0') {
        sprintf(logfile, "%s/%s%c.iscloc", revisedir, yyyymm, version);
        if ((logfp = fopen(logfile, "w")) == NULL) {
            fprintf(errfp, "Can't open log file %s\n", logfile);
            return 1;
        }
    }
    else if (logfile[0]) {
        if (streq(logfile, "stdout"))
            logfp = stdout;
        else if (streq(logfile, "stderr"))
            logfp = stderr;
        else {
            /* First > is meaningless - always write to logfile. */
            if (logfile[0] == '>') {
                i = 0;
                while ((logfile[i] = logfile[i+1])) i++;
            }
            /* But if there are two then want append. */
            if (logfile[0] == '>') {
                i = 0;
                while ((logfile[i] = logfile[i+1])) i++;
                if ((logfp = fopen(logfile, "a")) == NULL) {
                    fprintf(errfp, "Can't open log file %s\n", logfile);
                    return 1;
                }
            }
            else {
                if ((logfp = fopen(logfile, "w")) == NULL) {
                    fprintf(errfp, "Can't open log file %s\n", logfile);
                    return 1;
                }
            }
        }
    }
    else {
        fprintf(errfp, "No log file given\n");
        return 1;
    }
    return 0;
}
