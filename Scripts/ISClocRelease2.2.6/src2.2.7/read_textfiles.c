#include "iscloc.h"
extern int verbose;
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;

/*
 * Functions:
 *    read_config
 *    read_model
 *    read_instruction
 *    read_data_files
 */

/*
 *  Title:
 *     read_config
 *  Synopsis:
 *     Sets external variables to values read from configuration file
 *  Input Arguments:
 *     filename - pathname for the configuration file
 *  Return:
 *     0/1 on success/error
 *  Notes:
 *     Each line can have one configuration variable and its value, eg:
 *        max_iter = 16 (note that spaces around '=')
 *     Diagnostics from this function get written to stderr as log and error
 *        files are not yet open (their names are read in here).
 *  Called by:
 *     main
 */
int read_config(char *filename)
{
/*
 *  I/O
 */
    extern char logfile[FILENAMELEN];                       /* log filename */
    extern char errfile[FILENAMELEN];                     /* error filename */
    extern char isf_stafile[FILENAMELEN];  /* stafile when not read from db */
    extern int update_db;                       /* write result to database */
    extern char nextid_db[24];             /* get new ids from this account */
    extern int repid;    /* 'reporter' field for new hypocentres and assocs */
    extern char out_agency[VALLEN];/* author for new hypocentres and assocs */
    extern char in_agency[VALLEN];               /* author for input assocs */
/*
 *  TT
 */
    extern char ttime_table[VALLEN];              /* travel time table name */
    extern double default_depth;   /* used if seed hypocentre depth is NULL */
/*
 *  ETOPO
 */
    extern char etopofile[FILENAMELEN];          /* filename for ETOPO file */
    extern int etoponlon;           /* number of longitude samples in ETOPO */
    extern int etoponlat;            /* number of latitude samples in ETOPO */
    extern double etopores;                            /* cellsize in ETOPO */
/*
 *  iteration control
 */
    extern int min_iter;                        /* min number of iterations */
    extern int max_iter;                        /* max number of iterations */
    extern int min_phases;                          /* min number of phases */
    extern double sigmathres;            /* to exclude phases from solution */
    extern int do_correlated_errors;       /* account for correlated errors */
    extern int allow_damping;           /* allow damping in LSQR iterations */
    extern double confidence;         /* confidence level for uncertainties */
/*
 *  depth-phase depth solution requirements
 */
    extern int mindepthpha;  /* min # of depth phases for depth-phase depth */
    extern int ndepagency;      /* min # of agencies reporting depth phases */
/*
 *  depth resolution
 */
    extern double localdist;                   /* max local distance (degs) */
    extern int minlocalsta;      /* min number of stations within localdist */
    extern double spdist;                        /* max S-P distance (degs) */
    extern int min_s_p;                    /* min number of S-P phase pairs */
    extern int min_corepha;    /* min number of core reflections ([PS]c[PS] */
/*
 *  maximum allowable depth error for free depth solutions
 */
    extern double maxdeperror_shallow; /* max error for crustal free-depth */
    extern double maxdeperror_deep;       /* max error for deep free-depth */
/*
 *  magnitudes
 */
    extern char mbQ_table[VALLEN];            /* magnitude correction table */
    extern double body_mag_min_dist;        /* min delta for calculating mb */
    extern double body_mag_max_dist;        /* max delta for calculating mb */
    extern double surf_mag_min_dist;        /* min delta for calculating Ms */
    extern double surf_mag_max_dist;        /* max delta for calculating Ms */
    extern double body_mag_min_per;        /* min period for calculating mb */
    extern double body_mag_max_per;        /* max period for calculating mb */
    extern double surf_mag_min_per;        /* min period for calculating Ms */
    extern double surf_mag_max_per;        /* max period for calculating Ms */
    extern double surf_mag_max_depth;       /* max depth for calculating Ms */
    extern double pertol;         /* MSH period tolerance around MSZ period */
    extern double mag_range_warn_thresh;/* warn about outliers beyond range */
/*
 *  NA grid search parameters
 */
    extern int do_gridsearch;   /* perform grid search for initial location */
    extern double na_radius; /*s earch radius (degs) around prime epicentre */
    extern double na_deptol;       /* search radius (km) around prime depth */
    extern double na_ottol;   /* search radius (s) around prime origin time */
    extern double na_lpnorm;          /* p-value for norm to compute misfit */
    extern int na_itermax;                      /* max number of iterations */
    extern int na_nsamplei;                       /* size of initial sample */
    extern int na_nsample;                    /* size of subsequent samples */
    extern int na_ncells;                /* number of cells to be resampled */
    extern long iseed;                                /* random number seed */
/*
 *  agencies whose hypocenters not to be used in setting initial hypocentre
 */
    extern char nohypoagency[MAXBUF][AGLEN];
    extern int numnohypoagency;
/*
 *  RSTT parameters
 */
    extern char rstt_model[FILENAMELEN];    /* pathname for RSTT model file */
    extern int use_RSTT_PgLg;                /* use RSTT Pg/Lg predictions? */

    FILE *fp;
    char line[LINLEN];
    char par[PARLEN];
    char value[VALLEN], *s;
    int i;
/*
 *  Default parameter values
 */
    strcpy(logfile, "stdout");
    strcpy(errfile, "stderr");
    strcpy(isf_stafile, "");
    strcpy(nextid_db, "isc");
    strcpy(out_agency, "ISC");
    strcpy(in_agency, "ISC");
    strcpy(ttime_table, "ak135");
    strcpy(etopofile, "etopo5_bed_g_i2.bin");
    strcpy(mbQ_table, "GR");
    etoponlon = 4321;
    etoponlat = 2161;
    etopores = 0.0833333;
    default_depth = 0.;
    min_phases = 4;
    confidence = 90.;
    update_db = 0;
    min_iter = 4;
    max_iter = 20;
    sigmathres = 4.;
    do_correlated_errors = 1;
    allow_damping = 1;
    mindepthpha = 5;
    ndepagency = 2;
    localdist = 0.2;
    minlocalsta = 1;
    spdist = 3.;
    min_s_p = 5;
    min_corepha = 5;
    maxdeperror_shallow = 30;
    maxdeperror_deep = 60;
    do_gridsearch = 0;
    na_radius = 5.;
    na_deptol = 300.;
    na_ottol = 30.;
    na_lpnorm = 1.;
    na_itermax = 10;
    na_nsamplei = 1000;
    na_nsample = 100;
    na_ncells = 20;
    iseed = 5590L;
    body_mag_min_dist = 21.;
    body_mag_max_dist = 100.;
    surf_mag_min_dist = 20.;
    surf_mag_max_dist = 160.;
    body_mag_min_per  = 0.;
    body_mag_max_per  = 3.;
    surf_mag_min_per  = 10.;
    surf_mag_max_per  = 60.;
    surf_mag_max_depth = 60.;
    pertol = 5.;
    mag_range_warn_thresh = 2.2;
    numnohypoagency = 0;
    strcpy(rstt_model, "");
    use_RSTT_PgLg = 0;
/*
 *  Open configuration file
 */
    if ((fp = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "ABORT: cannot open %s\n", filename);
        return 1;
    }
/*
 *  Read configuration parameters.
 */
    while (fgets(line, LINLEN, fp)) {
/*
 *      skip blank lines or comments
 */
        if (sscanf(line, "%s = %s", par, value) < 2) continue;
        if (strncmp(  par, "#", 1) == 0) continue;
        if (strncmp(value, "#", 1) == 0) continue;
/*
 *      parse parameters
 */
        if (streq(par, "out_agency"))
            strcpy(out_agency, value);
        else if (streq(par, "in_agency"))
            strcpy(in_agency, value);
        else if (streq(par, "repid"))             repid = atoi(value);
        else if (streq(par, "update_db"))         update_db = atoi(value);
        else if (streq(par, "nextid_db"))         strcpy(nextid_db, value);
        else if (streq(par, "isf_stafile"))       strcpy(isf_stafile, value);
        else if (streq(par, "logfile"))           strcpy(logfile, value);
        else if (streq(par, "errfile"))           strcpy(errfile, value);
        else if (streq(par, "confidence"))        confidence = atof(value);
        else if (streq(par, "do_correlated_errors"))
            do_correlated_errors = atoi(value);
        else if (streq(par, "allow_damping"))     allow_damping = atoi(value);
/*
 *      limits
 */
        else if (streq(par, "min_phases"))        min_phases = atoi(value);
        else if (streq(par, "sigmathres"))        sigmathres = atof(value);
        else if (streq(par, "min_iter"))          min_iter = atoi(value);
        else if (streq(par, "max_iter"))          max_iter = atoi(value);
/*
 *      TT
 */
        else if (streq(par, "ttime_table"))
            strcpy(ttime_table, value);
/*
 *      ETOPO
 */
        else if (streq(par, "etopofile")) {
            strncpy(etopofile, value, VALLEN);
            etopofile[VALLEN-1] = '\0';
        }
        else if (streq(par, "etoponlon"))         etoponlon = atoi(value);
        else if (streq(par, "etoponlat"))         etoponlat = atoi(value);
        else if (streq(par, "etopores"))          etopores = atof(value);
/*
 *      depth control
 */
        else if (streq(par, "default_depth"))     default_depth = atof(value);
        else if (streq(par, "mindepthpha"))       mindepthpha = atoi(value);
        else if (streq(par, "ndepagency"))        ndepagency = atoi(value);
        else if (streq(par, "localdist"))         localdist = atof(value);
        else if (streq(par, "minlocalsta"))       minlocalsta = atoi(value);
        else if (streq(par, "spdist"))            spdist = atof(value);
        else if (streq(par, "min_s_p"))           min_s_p = atoi(value);
        else if (streq(par, "min_corepha"))       min_corepha = atoi(value);
        else if (streq(par, "maxdeperror_shallow"))
             maxdeperror_shallow = atof(value);
        else if (streq(par, "maxdeperror_deep"))
             maxdeperror_deep = atof(value);
/*
 *      magnitude determination
 */
        else if (streq(par, "mbQ_table")) {
            strncpy(mbQ_table, value, VALLEN);
            mbQ_table[VALLEN-1] = '\0';
        }
        else if (streq(par, "body_mag_min_dist"))
            body_mag_min_dist = atof(value);
        else if (streq(par, "body_mag_max_dist"))
            body_mag_max_dist = atof(value);
        else if (streq(par, "surf_mag_min_dist"))
            surf_mag_min_dist = atof(value);
        else if (streq(par, "surf_mag_max_dist"))
            surf_mag_max_dist = atof(value);
        else if (streq(par, "body_mag_min_per"))
            body_mag_min_per = atof(value);
        else if (streq(par, "body_mag_max_per"))
            body_mag_max_per = atof(value);
        else if (streq(par, "surf_mag_min_per"))
            surf_mag_min_per = atof(value);
        else if (streq(par, "surf_mag_max_per"))
            surf_mag_max_per = atof(value);
        else if (streq(par, "surf_mag_max_depth"))
            surf_mag_max_depth = atof(value);
        else if (streq(par, "pertol"))
            pertol = atof(value);
        else if (streq(par, "mag_range_warn_thresh"))
            mag_range_warn_thresh = atof(value);
/*
 *      NA search parameters
 */
        else if (streq(par, "do_gridsearch"))
            do_gridsearch = atoi(value);
        else if (streq(par, "na_radius"))       na_radius = atof(value);
        else if (streq(par, "na_deptol"))       na_deptol = atof(value);
        else if (streq(par, "na_ottol"))        na_ottol = atof(value);
        else if (streq(par, "na_lpnorm"))       na_lpnorm = atof(value);
        else if (streq(par, "na_itermax"))      na_itermax = atoi(value);
        else if (streq(par, "na_nsamplei"))     na_nsamplei = atoi(value);
        else if (streq(par, "na_nsample"))      na_nsample = atoi(value);
        else if (streq(par, "na_ncells"))       na_ncells = atoi(value);
        else if (streq(par, "iseed"))           iseed = atol(value);
/*
 *      agencies whose hypocenters not to be used in setting the initial hypo
 */
        else if (streq(par, "nohypo_agencies")) {
            i = 0;
            s = strtok(value, ", ");
            strcpy(nohypoagency[i++], s);
            while ((s = strtok(NULL, ", ")) != NULL)
                strcpy(nohypoagency[i++], s);
            numnohypoagency = i;
        }
/*
 *      RSTT parameters
 */
        else if (streq(par, "use_RSTT_PgLg"))   use_RSTT_PgLg = atoi(value);
        else if (streq(par, "rstt_model"))
            strcpy(rstt_model, value);
/*
 *      skip unrecognized parameters
 */
        else
            continue;
    }
    fclose(fp);
    return 0;
}

/*
 *  Title:
 *     read_model
 *  Synopsis:
 *     Sets external variables to values read from model file
 *  Input Arguments:
 *     filename - pathname for the model file
 *  Return:
 *     0/1 for success/error
 *  Notes:
 *     Each line can either have a parameter and its value, eg: moho = 33
 *        or be part of a list ended by a blank line eg phase_map
 *  Called by:
 *     read_data_files
 */
int read_model(char *filename)
{
    extern double moho;                                   /* depth of Moho */
    extern double conrad;                               /* depth of Conrad */
    extern double max_depth_km;                    /* max hypocenter depth */
    extern double psurfvel;       /* Pg velocity for elevation corrections */
    extern double ssurfvel;       /* Sg velocity for elevation corrections */
    extern PHASEMAP phase_map[MAXPHACODES];        /* IASPEI phase mapping */
    extern int phase_map_num;             /* number of phases in phase_map */
    extern PHASEWEIGHT phase_weight[MAXNUMPHA];         /* prior measerror */
    extern int phase_weight_num;       /* number of phases in phase_weight */
    extern char allowable_phase[MAXTTPHA][PHALEN];     /* allowable phases */
    extern int no_allowable_phase_num;       /* number of allowable phases */
    extern char firstPphase[MAXTTPHA][PHALEN];    /* first-arriving P list */
    extern int firstPphase_num;       /* number of first-arriving P phases */
    extern char firstSphase[MAXTTPHA][PHALEN];    /* first-arriving S list */
    extern int firstSphase_num;       /* number of first-arriving S phases */
    extern char firstPopt[MAXTTPHA][PHALEN];      /* optional first P list */
    extern int firstPopt_num;         /* number of optional first P phases */
    extern char firstSopt[MAXTTPHA][PHALEN];      /* optional first S list */
    extern int firstSopt_num;         /* number of optional first S phases */
    extern char no_resid_phase[MAXNUMPHA][PHALEN];   /* no-residual phases */
    extern int no_resid_phase_num;         /* number of no-residual phases */
    extern char mb_phase[MAXNUMPHA][PHALEN];         /* phases for mb calc */
    extern int mb_phase_num;                        /* number of mb phases */
    extern char ms_phase[MAXNUMPHA][PHALEN];         /* phases for Ms calc */
    extern int ms_phase_num;                        /* number of Ms phases */

    FILE *fp;
    char line[LINLEN];
    char par[PARLEN];
    char value[VALLEN];
    char rep_phase[VALLEN], phase[VALLEN];
    int numval, i;
/*
 *  Open model file or return an error.
 */
    if ((fp = fopen(filename, "r")) == NULL) {
        fprintf(errfp, "ABORT: cannot open %s\n", filename);
        fprintf(logfp, "ABORT: cannot open %s\n", filename);
        return 1;
    }
/*
 *  Read model file
 */
    while (fgets(line, LINLEN, fp)) {
/*
 *      skip blank lines or comments
 */
        if (sscanf(line, "%s", par) < 1) continue;
        if (strncmp(par, "#", 1) == 0)    continue;
/*
 *      parameter = value pairs
 */
        if (sscanf(line, "%s = %s", par, value) == 2) {
            if      (streq(par, "moho"))         moho = atof(value);
            else if (streq(par, "conrad"))       conrad = atof(value);
            else if (streq(par, "max_depth_km")) max_depth_km = atof(value);
            else if (streq(par, "psurfvel"))     psurfvel = atof(value);
            else if (streq(par, "ssurfvel"))     ssurfvel = atof(value);
            else continue;
            if (verbose > 1)
                fprintf(logfp, "    read_model: %s = %s\n", par, value);
        }
/*
 *      Phase map - end list with blank line.
 */
        else if (streq(par, "phase_map")) {
            i = 0;
            while (fgets(line, LINLEN, fp)) {
                if (sscanf(line, "%s", par) < 1)
                    break;
                if (sscanf(line, "%s%s", rep_phase, phase) < 2) {
                    strcpy(phase_map[i].rep_phase, rep_phase);
                    strcpy(phase_map[i].phase, "");
                }
                else {
                    if (strlen(rep_phase) > PHALEN || strlen(phase) > PHALEN) {
                        fprintf(errfp, "ABORT: phase too long %s\n", line);
                        fprintf(logfp, "ABORT: phase too long %s\n", line);
                        return 1;
                    }
                    strcpy(phase_map[i].rep_phase, rep_phase);
                    strcpy(phase_map[i].phase, phase);
                }
                if (++i > MAXPHACODES) {
                    fprintf(errfp, "ABORT: too many phases\n");
                    fprintf(logfp, "ABORT: too many phases\n");
                    return 1;
                }
            }
            phase_map_num = i;
            if (verbose > 1)
                fprintf(logfp, "    read %d codes into phase_map\n", i);
        }
/*
 *      List of allowable phase codes
 */
        else if (streq(par, "allowable_phases")) {
            i = 0;
            while (fgets(line, LINLEN, fp)) {
                if (sscanf(line, "%s", phase) < 1)
                    break;
                if (strlen(phase) > PHALEN) {
                    fprintf(errfp, "ABORT: phase too long %s\n", line);
                    fprintf(logfp, "ABORT: phase too long %s\n", line);
                    return 1;
                }
                strcpy(allowable_phase[i], phase);
                if (++i > MAXTTPHA) {
                    fprintf(errfp, "ABORT: too many no_resid_phase\n");
                    fprintf(logfp, "ABORT: too many no_resid_phase\n");
                    return 1;
                }
            }
            no_allowable_phase_num = i;
            if (verbose > 1)
                fprintf(logfp, "    read %d codes in allowable_phase\n", i);
        }
/*
 *      List of allowable first-arriving P phase codes
 */
        else if (streq(par, "allowable_first_P")) {
            i = 0;
            while (fgets(line, LINLEN, fp)) {
                if (sscanf(line, "%s", phase) < 1)
                    break;
                if (strlen(phase) > PHALEN) {
                    fprintf(errfp, "ABORT: phase too long %s\n", line);
                    fprintf(logfp, "ABORT: phase too long %s\n", line);
                    return 1;
                }
                strcpy(firstPphase[i], phase);
                if (++i > MAXTTPHA) {
                    fprintf(errfp, "ABORT: too many no_resid_phase\n");
                    fprintf(logfp, "ABORT: too many no_resid_phase\n");
                    return 1;
                }
            }
            firstPphase_num = i;
            if (verbose > 1)
                fprintf(logfp, "    read %d codes in allowable_first_P\n", i);
        }
/*
 *      List of optional first-arriving P phase codes
 */
        else if (streq(par, "optional_first_P")) {
            i = 0;
            while (fgets(line, LINLEN, fp)) {
                if (sscanf(line, "%s", phase) < 1)
                    break;
                if (strlen(phase) > PHALEN) {
                    fprintf(errfp, "ABORT: phase too long %s\n", line);
                    fprintf(logfp, "ABORT: phase too long %s\n", line);
                    return 1;
                }
                strcpy(firstPopt[i], phase);
                if (++i > MAXTTPHA) {
                    fprintf(errfp, "ABORT: too many no_resid_phase\n");
                    fprintf(logfp, "ABORT: too many no_resid_phase\n");
                    return 1;
                }
            }
            firstPopt_num = i;
            if (verbose > 1)
                fprintf(logfp, "    read %d codes in optional_first_P\n", i);
        }
/*
 *      List of allowable first-arriving S phase codes
 */
        else if (streq(par, "allowable_first_S")) {
            i = 0;
            while (fgets(line, LINLEN, fp)) {
                if (sscanf(line, "%s", phase) < 1)
                    break;
                if (strlen(phase) > PHALEN) {
                    fprintf(errfp, "ABORT: phase too long %s\n", line);
                    fprintf(logfp, "ABORT: phase too long %s\n", line);
                    return 1;
                }
                strcpy(firstSphase[i], phase);
                if (++i > MAXTTPHA) {
                    fprintf(errfp, "ABORT: too many no_resid_phase\n");
                    fprintf(logfp, "ABORT: too many no_resid_phase\n");
                    return 1;
                }
            }
            firstSphase_num = i;
            if (verbose > 1)
                fprintf(logfp, "    read %d codes in allowable_first_S\n", i);
        }
/*
 *      List of optional first-arriving P phase codes
 */
        else if (streq(par, "optional_first_S")) {
            i = 0;
            while (fgets(line, LINLEN, fp)) {
                if (sscanf(line, "%s", phase) < 1)
                    break;
                if (strlen(phase) > PHALEN) {
                    fprintf(errfp, "ABORT: phase too long %s\n", line);
                    fprintf(logfp, "ABORT: phase too long %s\n", line);
                    return 1;
                }
                strcpy(firstSopt[i], phase);
                if (++i > MAXTTPHA) {
                    fprintf(errfp, "ABORT: too many no_resid_phase\n");
                    fprintf(logfp, "ABORT: too many no_resid_phase\n");
                    return 1;
                }
            }
            firstSopt_num = i;
            if (verbose > 1)
                fprintf(logfp, "    read %d codes in optional_first_S\n", i);
        }
/*
 *      List of phase codes that shouldn't have residuals calculated
 */
        else if (streq(par, "no_resid_phase")) {
            i = 0;
            while (fgets(line, LINLEN, fp)) {
                if (sscanf(line, "%s", phase) < 1)
                    break;
                if (strlen(phase) > PHALEN) {
                    fprintf(errfp, "ABORT: phase too long %s\n", line);
                    fprintf(logfp, "ABORT: phase too long %s\n", line);
                    return 1;
                }
                strcpy(no_resid_phase[i], phase);
                if (++i > MAXNUMPHA) {
                    fprintf(errfp, "ABORT: too many no_resid_phase\n");
                    fprintf(logfp, "ABORT: too many no_resid_phase\n");
                    return 1;
                }
            }
            no_resid_phase_num = i;
            if (verbose > 1)
                fprintf(logfp, "    read %d codes in no_resid_phase\n", i);
        }
/*
 *      Phase weight structure
 */
        else if (streq(par, "phase_weight")) {
            i = 0;
            while (fgets(line, LINLEN, fp)) {
                if (sscanf(line, "%s", par) < 1)
                    break;
                numval = sscanf(line, "%s%lf%lf%lf",
                    phase, &phase_weight[i].delta1, &phase_weight[i].delta2,
                    &phase_weight[i].measurement_error);
                if (numval < 4) {
                    fprintf(errfp, "ABORT: incomplete entry %s\n", line);
                    fprintf(logfp, "ABORT: incomplete entry %s\n", line);
                    return 1;
                }
                if (strlen(phase) > PHALEN) {
                    fprintf(errfp, "ABORT: phase too long %s\n", line);
                    fprintf(logfp, "ABORT: phase too long %s\n", line);
                    return 1;
                }
                strcpy(phase_weight[i].phase, phase);
                if (++i > MAXNUMPHA) {
                    fprintf(errfp, "ABORT: too many phases\n");
                    fprintf(logfp, "ABORT: too many phases\n");
                    return 1;
                }
            }
            phase_weight_num = i;
            if (verbose > 1)
                fprintf(logfp, "    read %d lines into phase_weight\n",i);
        }
/*
 *      List of phase codes that contribute to mb calculation.
 */
        else if (streq(par, "mb_phase")) {
            i = 0;
            while (fgets(line, LINLEN, fp)) {
                if (sscanf(line, "%s", phase) < 1)
                    break;
                if (strlen(phase) > PHALEN) {
                    fprintf(errfp, "ABORT: phase too long %s\n", line);
                    fprintf(logfp, "ABORT: phase too long %s\n", line);
                    return 1;
                }
                strcpy(mb_phase[i], phase);
                if (++i > MAXNUMPHA) {
                    fprintf(errfp, "ABORT: too many mb_phase\n");
                    fprintf(logfp, "ABORT: too many mb_phase\n");
                    return 1;
                }
            }
            mb_phase_num = i;
            if (verbose > 1)
                fprintf(logfp, "    read %d codes in mb_phase\n", i);
        }
/*
 *      List of phase codes that contribute to Ms calculation.
 */
        else if (streq(par, "ms_phase")) {
            i = 0;
            while (fgets(line, LINLEN, fp)) {
                if (sscanf(line, "%s", phase) < 1)
                    break;
                if (strlen(phase) > PHALEN) {
                    fprintf(errfp, "ABORT: phase too long %s\n", line);
                    fprintf(logfp, "ABORT: phase too long %s\n", line);
                    return 1;
                }
                strcpy(ms_phase[i], phase);
                if (++i > MAXNUMPHA) {
                    fprintf(errfp, "ABORT: too many ms_phase\n");
                    fprintf(logfp, "ABORT: too many ms_phase\n");
                    return 1;
                }
            }
            ms_phase_num = i;
            if (verbose > 1)
                fprintf(logfp, "    read %d codes in ms_phase\n", i);
        }
        else
            continue;
    }
    fclose(fp);
    return 0;
}

/*
 *   Title:
 *      read_instruction
 *   Synopsis:
 *      Parses a line of instructions for evid and options
 *      Parses an instruction line for an evid and its instructions.
 *      Instructions can either be given in the command line or
 *      read from an instruction file. An instruction line is expected
 *      for every event in the form:
 *          isc_evid [par=value [par=value ...]]
 *      where isc_evid is the event identifier, and the par=value pairs
 *      (no space around the equal sign!) denote the optional instruction
 *      name and value pairs.
 *   Input Arguments:
 *      instruction - string containing a single line of instructions
 *      ep          - pointer to event info
 *   Return:
 *      0/1 on success/error
 *   Notes:
 *      One or more options may be given in any order in parameter=value
 *          format separated by white space.
 *      Possible options:
 *          verbose           - verbose level
 *          depth             - an agency code or a number
 *          lat,lon           - numbers (must have both)
 *          time              - an agency code or a number
 *          fix_depth         - an agency code or a number or nothing
 *          fix_location      - an agency code or nothing
 *          fix_time          - an agency code or a number or nothing
 *          fix_hypo          - an agency code or a number or nothing
 *          fix_depth_depdp   - presence sets flag to 1
 *          fix_depth_default - presence sets flag to 1
 *          indb              - read data from this DB account
 *          outdb             - write results to this DB account
 *          isf_infile        - read ISF input from here
 *          isf_outfile       - write ISF output to here
 *          write_gridsearch_results - write NA results to file
 *      The options below can override external parameters in the config file:
 *          ndepagency        - min # of agencies reporting depth/core phases
 *          mindepthpha       - min # of depth phases for depth-phase depth
 *          min_corepha       - min number of core reflections ([PS]c[PS]
 *          localdist         - max local distance (degs)
 *          minlocalsta       - min number of stations within localdist
 *          spdist            - max S-P distance (degs)
 *          min_s_p           - min number of S-P phase pairs
 *          update_db         - write results to database?
 *          do_correlated_errors - account for correlated error structure?
 *          do_gridsearch     - perform initial grid search?
 *          iseed             - random number initial seed
 *          na_radius         - search radius [deg] around initial epicentre
 *          na_ottol          - search radius [s] around initial origin time
 *          na_deptol         - search radius [km] around initial depth
 *          na_nsamplei       - size of initial sample
 *          na_nsample        - size of subsequent samples
 *          na_ncells         - number of cells to be resampled
 *          na_itermax        - max number of iterations
 *          isf_stafile       - file of station coordinates
 *          pertol            - MSH period tolerance around MSZ period
 *          rstt_model        - pathname for RSTT model file
 *          use_RSTT_PgLg     - use RSTT Pg/Lg predictions?
 *  Called by:
 *     main
 *  Calls:
 *     read_time
 */
int read_instruction(char *instruction, EVREC *ep, int isf)
{
    extern int do_correlated_errors;       /* account for correlated errors */
    extern int do_gridsearch;         /* perform NA search for initial hypo */
    extern double na_radius; /* search radius (degs) around prime epicentre */
    extern double na_deptol;       /* search radius (km) around prime depth */
    extern double na_ottol;   /* search radius (s) around prime origin time */
    extern int na_itermax;                      /* max number of iterations */
    extern int na_nsamplei;                       /* size of initial sample */
    extern int na_nsample;                    /* size of subsequent samples */
    extern int na_ncells;                /* number of cells to be resampled */
    extern int na_itermax;                      /* max number of iterations */
    extern int write_gridsearch_results;
    extern int mindepthpha;  /* min # of depth phases for depth-phase depth */
    extern int ndepagency;      /* min # of agencies reporting depth phases */
    extern double localdist;                   /* max local distance (degs) */
    extern int minlocalsta;      /* min number of stations within localdist */
    extern double spdist;                        /* max S-P distance (degs) */
    extern int min_s_p;                    /* min number of S-P phase pairs */
    extern int min_corepha;    /* min number of core reflections ([PS]c[PS] */
    extern long iseed;
    extern double pertol;         /* MSH period tolerance around MSZ period */
    extern int update_db;                       /* write result to database */
    extern char indb[24];         /* read data from this DB account, if any */
    extern char outdb[24];      /* write results to this DB account, if any */
    extern char isf_stafile[FILENAMELEN];  /* stafile when not read from db */
    extern char isf_infile[FILENAMELEN];              /* input ISF filename */
    extern char isf_outfile[FILENAMELEN];            /* output ISF filename */
    extern char rstt_model[FILENAMELEN];    /* pathname for RSTT model file */
    extern int use_RSTT_PgLg;                /* use RSTT Pg/Lg predictions? */
    char opt[MAXOPT][PARLEN];
    char val[MAXOPT][VALLEN];
    char *remnant;
    int i, j, k;
    int value_option, numopt;
/*
 *  Set default values for options.
 */
    ep->evid               = NULLVAL;
    ep->depth_agency[0]    = '\0';
    ep->location_agency[0] = '\0';
    ep->time_agency[0]     = '\0';
    ep->hypo_agency[0]     = '\0';
    ep->start_depth        = NULLVAL;
    ep->start_lat          = NULLVAL;
    ep->start_lon          = NULLVAL;
    ep->start_time         = NULLVAL;
    ep->surface_fix        = 0;
    ep->time_fix           = 0;
    ep->depth_fix          = 0;
    ep->depth_fix_editor   = 0;
    ep->epi_fix            = 0;
    ep->hypo_fix           = 0;
    ep->fix_depth_default  = 0;
    ep->fix_depth_depdp    = 0;
    ep->fix_depth_median   = 0;
/*
 *  Remove leading white space
 */
    i = 0;
    while (instruction[i] == ' ' || instruction[i] == '\t') { i++; }
/*
 *  Read in unknown number of options.
 */
    k = 0;
    while (instruction[i] != '\n') {
/*
 *      Option name
 */
        j = 0;
        value_option = 1;
        while (instruction[i] != '\n') {
            if (instruction[i] == '=') {
                i++;
                break;
            }
            if (instruction[i] == ' ') {
                value_option = 0;
                i++;
                break;
            }
            opt[k][j++] = instruction[i++];
        }
        opt[k][j] = '\0';
/*
 *      Option value
 */
        j = 0;
        if (value_option)
            while (instruction[i] != '\n') {
                if (instruction[i] == ' ' ) {
                    i++;
                    break;
                }
                val[k][j++] = instruction[i++];
            }
        val[k][j] = '\0';
        k++;
    }
    numopt = k;
/*
 *  Parse options and update event structure.
 */
    for (i = 0; i < numopt; i++) {
/*
 *      A parameter without a value is interpreted as evid.
 */
        if (isdigit(opt[i][0])) {
            if (ep->evid != NULLVAL || isf) {
                fprintf(errfp, "ABORT: parameter with no value %s\n", val[i]);
                fprintf(errfp, "ABORT: parameter with no value %s\n", val[i]);
                return 1;
            }
            ep->evid = atoi(opt[i]);
        }
/*
 *      set initial depth to (number or agency)
 */
        else if (streq(opt[i], "depth")) {
            if (isdigit(val[i][0])) {
                ep->start_depth = strtod(val[i], &remnant);
                if (remnant[0]) {
                    fprintf(errfp, "ABORT: bad depth %s\n", val[i]);
                    fprintf(logfp, "ABORT: bad depth %s\n", val[i]);
                    return 1;
                }
            }
            else
                strcpy(ep->depth_agency, val[i]);
        }
/*
 *      fix depth to (number or agency)
 */
        else if (streq(opt[i], "fix_depth")) {
            if (isdigit(val[i][0])) {
                ep->start_depth = strtod(val[i], &remnant);
                if (remnant[0]) {
                    fprintf(errfp, "ABORT: bad fix_depth %s\n",
                            val[i]);
                    fprintf(logfp, "ABORT: bad fix_depth %s\n",
                            val[i]);
                    return 1;
                }
                ep->depth_fix_editor = 1;
            }
            else
                strcpy(ep->depth_agency, val[i]);
            ep->depth_fix = 1;
        }
/*
 *      set initial latitude to (number or agency)
 */
        else if (streq(opt[i], "lat")) {
            if (isdigit(val[i][0]) || val[i][0] == '-') {
                ep->start_lat = strtod(val[i], &remnant);
                if (remnant[0]) {
                    fprintf(errfp, "ABORT: bad lat %s\n", val[i]);
                    fprintf(logfp, "ABORT: bad lat %s\n", val[i]);
                    return 1;
                }
            }
            else
                strcpy(ep->location_agency, val[i]);
        }
/*
 *      set initial longitude to (number or agency or none)
 */
        else if (streq(opt[i], "lon")) {
            if (isdigit(val[i][0]) || val[i][0] == '-') {
                ep->start_lon = strtod(val[i], &remnant);
                if (remnant[0]) {
                    fprintf(errfp, "ABORT: bad lon %s\n", val[i]);
                    fprintf(logfp, "ABORT: bad lon %s\n", val[i]);
                    return 1;
                }
            }
            else
                strcpy(ep->location_agency, val[i]);
        }
/*
 *      fix epicenter to agency
 */
        else if (streq(opt[i], "fix_location")) {
            strcpy(ep->location_agency, val[i]);
            ep->epi_fix = 1;
        }
/*
 *      set initial origin time to (yyyy-mm-dd_hh:mi:ss.sss or agency)
 */
        else if (streq(opt[i], "time")) {
            if (isdigit(val[i][0])) {
                if ((ep->start_time = read_time(val[i])) == 0) {
                    fprintf(errfp, "ABORT: bad time %s\n", val[i]);
                    fprintf(logfp, "ABORT: bad time %s\n", val[i]);
                    return 1;
                }
            }
            else
                strcpy(ep->time_agency, val[i]);
        }
/*
 *      fix origin time to (yyyy-mm-dd_hh:mi:ss.sss or agency)
 */
        else if (streq(opt[i], "fix_time")) {
            if (isdigit(val[i][0])) {
                if ((ep->start_time = read_time(val[i])) == 0) {
                    fprintf(errfp, "ABORT: bad time %s\n", val[i]);
                    fprintf(logfp, "ABORT: bad time %s\n", val[i]);
                    return 1;
                }
            }
            else
                strcpy(ep->time_agency, val[i]);
            ep->time_fix = 1;
        }
/*
 *      fix hypocenter to agency
 */
        else if (streq(opt[i], "fix_hypo"))
            strcpy(ep->hypo_agency, val[i]);
/*
 *      fix depth to depth-phase depth - obsolete
 *
 *      else if (streq(opt[i], "fix_depth_depdp")) {
 *          ep->fix_depth_depdp = atoi(val[i]);
 *          if (ep->fix_depth_depdp)
 *              ep->depth_fix = 1;
 *      }
 */
/*
 *      fix depth to default region-dependent depth
 */
        else if (streq(opt[i], "fix_depth_default")) {
            ep->fix_depth_default = atoi(val[i]);
            if (ep->fix_depth_default)
                ep->depth_fix = 1;
        }
/*
 *      fix depth to median of reported depths
 */
        else if (streq(opt[i], "fix_depth_median")) {
            ep->fix_depth_median = atoi(val[i]);
            if (ep->fix_depth_median)
                ep->depth_fix = 1;
        }
/*
 *      depth resolution
 */
        else if (streq(opt[i], "mindepthpha"))
            mindepthpha = atoi(val[i]);
        else if (streq(opt[i], "ndepagency"))
            ndepagency = atoi(val[i]);
        else if (streq(opt[i], "localdist")) {
            localdist = strtod(val[i], &remnant);
            if (remnant[0]) {
                fprintf(errfp, "ABORT: bad localdist %s\n", val[i]);
                fprintf(logfp, "ABORT: bad localdist %s\n", val[i]);
                return 1;
            }
        }
        else if (streq(opt[i], "spdist")) {
            spdist = strtod(val[i], &remnant);
            if (remnant[0]) {
                fprintf(errfp, "ABORT: bad spdist %s\n", val[i]);
                fprintf(logfp, "ABORT: bad spdist %s\n", val[i]);
                return 1;
            }
        }
        else if (streq(opt[i], "minlocalsta"))
            minlocalsta = atoi(val[i]);
        else if (streq(opt[i], "min_s_p"))
            min_s_p = atoi(val[i]);
        else if (streq(opt[i], "min_corepha"))
            min_corepha = atoi(val[i]);
/*
 *      account for correlated error structure [0/1]?
 */
        else if (streq(opt[i], "do_correlated_errors"))
            do_correlated_errors = atoi(val[i]);
/*
 *      set verbose level
 */
        else if (streq(opt[i], "verbose"))
            verbose = atoi(val[i]);
/*
 *      perform NA grid search [0/1]?
 */
        else if (streq(opt[i], "do_gridsearch"))
            do_gridsearch = atoi(val[i]);
/*
 *      NA search radius [degrees] around initial epicentre
 */
        else if (streq(opt[i], "na_radius")) {
            na_radius = strtod(val[i], &remnant);
            if (remnant[0]) {
                fprintf(errfp, "ABORT: bad na_radius %s\n", val[i]);
                fprintf(logfp, "ABORT: bad na_radius %s\n", val[i]);
                return 1;
            }
        }
/*
 *      NA search radius for depth [km] around initial depth
 */
        else if (streq(opt[i], "na_deptol")) {
            na_deptol = strtod(val[i], &remnant);
            if (remnant[0]) {
                fprintf(errfp, "ABORT: bad na_deptol %s\n", val[i]);
                fprintf(logfp, "ABORT: bad na_deptol %s\n", val[i]);
                return 1;
            }
        }
/*
 *      NA search radius for origin time [s] around initial OT
 */
        else if (streq(opt[i], "na_ottol")) {
            na_ottol = strtod(val[i], &remnant);
            if (remnant[0]) {
                fprintf(errfp, "ABORT: bad na_ottol %s\n", val[i]);
                fprintf(logfp, "ABORT: bad na_ottol %s\n", val[i]);
                return 1;
            }
        }
/*
 *      number of initial samples in NA search
 */
        else if (streq(opt[i], "na_nsamplei"))
            na_nsamplei = atoi(val[i]);
/*
 *      number of subsequent samples in NA search
 */
        else if (streq(opt[i], "na_nsample"))
            na_nsample = atoi(val[i]);
/*
 *      number of cells to be resampled in NA search
 */
        else if (streq(opt[i], "na_ncells"))
            na_ncells = atoi(val[i]);
/*
 *      max number of iterations in NA search
 */
        else if (streq(opt[i], "na_itermax"))
            na_itermax = atoi(val[i]);
/*
 *      write NA search results to file [0/1]?
 */
        else if (streq(opt[i], "write_gridsearch_results"))
            write_gridsearch_results = atoi(val[i]);
/*
 *      initial random number seed for NA search
 */
        else if (streq(opt[i], "iseed")) {
            iseed = (long)strtod(val[i], &remnant);
            if (remnant[0]) {
                fprintf(errfp, "read_instruction: bad iseed %s\n", val[i]);
                fprintf(logfp, "read_instruction: bad iseed %s\n", val[i]);
                return 1;
            }
        }
/*
 *      MS_H period tolerance around MS_Z period
 */
        else if (streq(opt[i], "pertol")) {
            pertol = strtod(val[i], &remnant);
            if (remnant[0]) {
                fprintf(errfp, "ABORT: bad pertol %s\n", val[i]);
                fprintf(logfp, "ABORT: bad pertol %s\n", val[i]);
                return 1;
            }
        }
/*
 *      ISF input/output
 */
        else if (streq(opt[i], "isf_stafile"))
            strcpy(isf_stafile, val[i]);
        else if (streq(opt[i], "isf_infile"))
            strcpy(isf_infile, val[i]);
        else if (streq(opt[i], "isf_outfile"))
            strcpy(isf_outfile, val[i]);
/*
 *      set input DB account
 */
        else if (streq(opt[i], "indb")) {
            strcpy(indb, val[i]);
            strcat(indb, ".");
        }
/*
 *      set output DB account
 */
        else if (streq(opt[i], "outdb")) {
            strcpy(outdb, val[i]);
            strcat(outdb, ".");
        }
/*
 *      update DB [0/1]?
 */
        else if (streq(opt[i], "update_db"))
            update_db = atoi(val[i]);
/*
 *      use RSTT Pg/Lg predictions [0/1]?
 */
        else if (streq(opt[i], "use_RSTT_PgLg"))
            use_RSTT_PgLg = atoi(val[i]);
/*
 *      set RSTT model
 */
        else if (streq(opt[i], "rstt_model"))
            strcpy(rstt_model, val[i]);
/*
 *      instruction not recognised
 */
        else {
            fprintf(errfp, "read_instruction: unknown option %s\n", opt[i]);
            fprintf(logfp, "read_instruction: unknown option %s\n", opt[i]);
        }
        if (verbose > 1)
            fprintf(logfp, "    read_instruction: %d %s %s\n",
                    i, opt[i], val[i]);
    }
/*
 *
 *  sanity checks
 *
 *
 *  must be an evid or a filename somewhere on line
 */
    if (ep->evid == NULLVAL && !isf) {
        fprintf(errfp, "ABORT: no evid is given!\n%s\n", instruction);
        fprintf(logfp, "ABORT: no evid is given!\n %s\n", instruction);
        return 1;
    }
/*
 *  check for valid ISF input if it is expected
 */
    if (isf) {
        if (streq(isf_stafile, "")) {
            fprintf(errfp, "ABORT: No ISF station file is given!\n");
            fprintf(logfp, "ABORT: No ISF station file is given!\n");
            return 1;
        }
        if (streq(isf_infile, "")) {
            fprintf(errfp, "ABORT: No ISF input file is given!\n");
            fprintf(logfp, "ABORT: No ISF input file is given!\n");
            return 1;
        }
    }
/*
 *  Checks to see that not fixing something two ways at once.
 */
    if (ep->start_lat != NULLVAL || ep->start_lon != NULLVAL) {
        if (ep->start_lat == NULLVAL || ep->start_lon == NULLVAL) {
            fprintf(errfp, "ABORT: must set both lat and lon\n");
            fprintf(logfp, "ABORT: must set both lat and lon\n");
            return 1;
        }
        if (ep->location_agency[0]) {
            fprintf(errfp, "ABORT: location set twice\n");
            fprintf(logfp, "ABORT: location set twice\n");
            return 1;
        }
    }
    if (verbose > 1)
        fprintf(logfp, "    read_instruction: evid: %d\n", ep->evid);
    return 0;
}

/*
 *  Title:
 *     read_data_files
 *  Synopsis:
 *     read data files from config directory
 *        $QETC/iscloc/<ttime_table>_model.txt
 *        $QETC/<ttime_table>/[*].tab
 *        $QETC/ak135/ELCOR.dat
 *        $QETC/topo/etopo5_bed_g_i2.bin
 *        $QETC/FlinnEngdahl/FE.dat
 *        $QETC/FlinnEngdahl/default.depth0.5.grid
 *        $QETC/FlinnEngdahl/grn_default_depth.ak135.dat
 *        $QETC/variogram/variogram.model
 *        $QETC/magnitude/<mbQ_table>.dat
 *  Input Arguments:
 *     configdir - pathname for config directory
 *  Output Arguments:
 *     ismbQ        - use magnitude attenuation table? (0/1)
 *     mbQp         - pointer to MAGQ structure
 *     fep          - pointer to FE structure
 *     gres         - grid spacing in default depth grid
 *     ngrid        - number of grid points in default depth grid
 *     depthgrid    - pointer to default depth grid (lat, lon, depth)
 *     num_ecphases - number of phases with ellipticity correction
 *     ec           - ellipticity correction coefs
 *     tt_tables    - pointer to travel-time tables
 *     variogramp   - pointer to variogram structure
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     main
 *  Calls:
 *     read_model, read_FlinnEngdahl, read_default_depth_region,
 *     read_etopo1, read_default_depth_grid, read_elcor_tbl,
 *     read_tt_tables, read_magQ, read_variogram, free_fe, free_elcor_tbl,
 *     free_tt_tbl, Free, free_matrix, free_i2matrix
 */
int read_data_files(char *configdir, int *ismbQ, MAGQ *mbQp,
                    FE *fep, double **grn_depth, double *gres, int *ngrid,
                    double ***depthgrid, short int ***topo,
                    int *num_ecphases, EC_COEF *ec[],
                    TT_TABLE *tt_tables[], VARIOGRAM *variogramp)
{
    extern char etopofile[FILENAMELEN];          /* filename for ETOPO file */
    extern char ttime_table[VALLEN];              /* travel time table name */
    extern char mbQ_table[VALLEN];            /* magnitude correction table */
    char filename[FILENAMELEN];
    char dirname[FILENAMELEN];
    double *gd = (double *)NULL;
/*
 *  Read model file from configdir/iscloc directory
 */
    if (streq(ttime_table, "jb")) {
        sprintf(filename, "%s/iscloc/jb_model.txt", configdir);
        sprintf(dirname, "%s/jb", configdir);
        fprintf(logfp, "model=jb\n");
    }
    else if (streq(ttime_table, "ak135")) {
        sprintf(filename, "%s/iscloc/ak135_model.txt", configdir);
        sprintf(dirname, "%s/ak135", configdir);
        fprintf(logfp, "model=ak135\n");
    }
    else {
        fprintf(errfp, "Bad ttime_table %s in config.txt\n", ttime_table);
        return 1;
    }
    fprintf(logfp, "    read_model: %s\n", filename);
    if (read_model(filename))
        return 1;
/*
 *  Read Flinn-Engdahl region numbers and default depth files
 *  from configdir/FlinnEngdahl directory
 */
    sprintf(filename, "%s/FlinnEngdahl/FE.dat", configdir);
    fprintf(logfp, "    read_FlinnEngdahl: %s\n", filename);
    if (read_FlinnEngdahl(filename, fep))
        return 1;
    sprintf(filename, "%s/FlinnEngdahl/grn_default_depth.%s.dat",
            configdir, ttime_table);
    fprintf(logfp, "    read_default_depth_region: %s\n", filename);
    if ((gd = read_default_depth_region(filename)) == NULL) {
        fprintf(errfp, "Cannot read default GRN depths!\n");
        free_fe(fep);
        return 1;
    }
    sprintf(filename, "%s/FlinnEngdahl/default.depth0.5.grid", configdir);
    fprintf(logfp, "    read_default_depth_grid: %s\n", filename);
    if ((*depthgrid = read_default_depth_grid(filename, gres, ngrid)) == NULL) {
        fprintf(errfp, "Cannot read default depth grid!\n");
        free_fe(fep); Free(gd);
        return 1;
    }
/*
 *  Read travel-time tables
 */
    if (streq(ttime_table, "ak135")) {
/*
 *      read ellipticity correction file
 */
        sprintf(filename, "%s/ELCOR.dat", dirname);
        fprintf(logfp, "    read_elcor_tbl: %s\n", filename);
        if ((*ec = read_elcor_tbl(num_ecphases, filename)) == NULL) {
            fprintf(errfp, "Cannot read ellipticity correction file %s\n",
                    filename);
            free_fe(fep); Free(gd);
            free_matrix(*depthgrid);
            return 1;
        }
/*
 *      read ak135 TT tables
 */
        fprintf(logfp, "    read_tt_tables: %s\n", dirname);
        if ((*tt_tables = read_tt_tables(dirname)) == NULL) {
            fprintf(errfp, "Cannot read ak135 TT tables!\n");
            if (*ec != NULL) free_elcor_tbl(*ec, *num_ecphases);
            free_fe(fep); Free(gd);
            free_matrix(*depthgrid);
            return 1;
        }
    }
    if (streq(ttime_table, "jb")) {
/*
 *      read JB TT tables
 */
        fprintf(logfp, "    read_tt_tables: %s\n", dirname);
        if ((*tt_tables = read_tt_tables(dirname)) == NULL) {
            fprintf(errfp, "Cannot read JB TT tables!\n");
            free_fe(fep); Free(gd);
            free_matrix(*depthgrid);
            return 1;
        }
    }
/*
 *  read ETOPO1 file for bounce point corrections from configdir/topo directory
 */
    sprintf(filename, "%s/topo/%s", configdir, etopofile);
    fprintf(logfp, "    read_etopo1: %s\n", filename);
    if ((*topo = read_etopo1(filename)) == NULL) {
        fprintf(errfp, "Cannot read ETOPO file %s\n", filename);
        if (*tt_tables != NULL) free_tt_tbl(*tt_tables);
        if (*ec != NULL) free_elcor_tbl(*ec, *num_ecphases);
        free_fe(fep); Free(gd);
        free_matrix(*depthgrid);
        return 1;
    }
/*
 *  Read magnitude attenuation Q(d, h) table from configdir/magnitude directory
 */
    if (strcmp(mbQ_table, "none")) {
        sprintf(filename, "%s/magnitude/%smbQ.dat", configdir, mbQ_table);
        fprintf(logfp, "    read_magQ: %s\n", filename);
        if (read_magQ(filename, mbQp)) {
            fprintf(errfp, "Cannot read magnitude Q file %s\n", filename);
            if (*tt_tables != NULL) free_tt_tbl(*tt_tables);
            if (*ec != NULL) free_elcor_tbl(*ec, *num_ecphases);
            free_fe(fep); Free(gd);
            free_matrix(*depthgrid);
            free_i2matrix(*topo);
            return 1;
        }
        if (strstr(mbQ_table, "GR")) *ismbQ = 1;
        if (strstr(mbQ_table, "VC")) *ismbQ = 2;
        if (strstr(mbQ_table, "MB")) *ismbQ = 2;
    }
/*
 *  Read generic variogram model from configdir/variogram directory
 */
    sprintf(filename, "%s/variogram/variogram.model", configdir);
    fprintf(logfp, "    read_variogram: %s\n", filename);
    if (read_variogram(filename, variogramp)) {
        fprintf(errfp, "Cannot read variogram file %s\n", filename);
        if (*tt_tables != NULL) free_tt_tbl(*tt_tables);
        if (*ec != NULL) free_elcor_tbl(*ec, *num_ecphases);
        free_fe(fep); Free(gd);
        free_matrix(*depthgrid);
        free_i2matrix(*topo);
        if (*ismbQ) {
            Free(mbQp->deltas);
            Free(mbQp->depths);
            free_matrix(mbQp->q);
        }
        return 1;
    }
    *grn_depth = gd;
    return 0;
}

