#include "iscloc.h"
extern int verbose;
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;
extern int phaseTT_num;
extern char phaseTT[MAXTTPHA][PHALEN];
extern char ttime_table[];                                  /* model name */
extern double psurfvel;          /* Pg velocity for elevation corrections */
extern double ssurfvel;          /* Sg velocity for elevation corrections */
extern double max_depth_km;       /* max hypocenter depth from model file */
extern char no_resid_phase[MAXNUMPHA][PHALEN];         /* from model file */
extern int no_resid_phase_num;         /* number of 'non-residual' phases */
extern int etoponlon;           /* number of longitude samples in ETOPO */
extern int etoponlat;            /* number of latitude samples in ETOPO */
extern double etopores;                            /* cellsize in ETOPO */

/*
 * Functions:
 *    read_tt_tables
 *    free_tt_tbl
 *    read_etopo1
 *    get_phase_index
 *    get_tt
 *    calc_resid
 *    read_ttime
 *    topcor
 */

/*
 * Local functions:
 *    find_etopo1
 *    correct_time
 *    get_bounce_corr
 *    get_elev_corr
 *    last_lag
 *    calc_geoid_corr
 *    height_above_mean_sphere
 *    timeres
 */
static double find_etopo1(double lat, double lon, short int **topo);
static void correct_ttime(SOLREC *sp, PHAREC *pp, EC_COEF *ec,
                          short int **topo);
static double get_bounce_corr(SOLREC *sp, PHAREC *pp,
                              short int **topo, double *tcorw);
static double get_elev_corr(PHAREC *pp);
static int last_lag(char phase[]);
static double calc_geoid_corr(double lat, PHAREC *pp);
static double height_above_mean_sphere(double lat);
static double timeres(PHAREC *pp, int all, SOLREC *sp, EC_COEF *ec,
                      TT_TABLE *tt_tables, short int **topo, int iszderiv);

/*
 *  Title:
 *     read_tt_tables
 *  Synopsis:
 *     Read travel-time tables from files in dirname.
 *  Input Arguments:
 *     dirname - directory pathname for TT tables
 *  Return:
 *     tt_tables - pointer to TT_TABLE structure or NULL on error
 *  Called by:
 *     read_data_files
 *  Calls:
 *     skipcomments, free_tt_tbl, alloc_matrix
 */
TT_TABLE *read_tt_tables(char *dirname)
{
    FILE *fp;
    TT_TABLE *tt_tables = (TT_TABLE *)NULL;
    char fname[MAXBUF], buf[LINLEN], *s;
    int ndists = 0, ndepths = 0, i, j, k, m, ind = 0, isdepthphase = 0;
/*
 *  memory allocation
 */
    tt_tables = (TT_TABLE *)calloc(phaseTT_num, sizeof(TT_TABLE));
    if (tt_tables == NULL) {
        fprintf(logfp, "read_tt_tables: cannot allocate memory\n");
        fprintf(errfp, "read_tt_tables: cannot allocate memory\n");
        errorcode = 1;
        return (TT_TABLE *) NULL;
    }
/*
 *  read TT table files
 *      phaseTT_num and phaseTT are specified in iscloc.h
 */
    for (ind = 0; ind < phaseTT_num; ind++) {
/*
 *      initialize tt_tables for this phase
 */
        isdepthphase = 0;
        if (phaseTT[ind][0] == 'p' || phaseTT[ind][0] == 's')
            isdepthphase = 1;
        strcpy(tt_tables[ind].phase, phaseTT[ind]);
        tt_tables[ind].ndel = 0;
        tt_tables[ind].ndep = 0;
        tt_tables[ind].deltas = (double *)NULL;
        tt_tables[ind].depths = (double *)NULL;
        tt_tables[ind].tt = (double **)NULL;
        tt_tables[ind].dtdd = (double **)NULL;
        tt_tables[ind].dtdh = (double **)NULL;
        tt_tables[ind].bpdel = (double **)NULL;
/*
 *      open TT table file for this phase
 */
        if (isdepthphase)
            sprintf(fname, "%s/%s.little%s.tab",
                    dirname, ttime_table, phaseTT[ind]);
        else
            sprintf(fname, "%s/%s.%s.tab",
                    dirname, ttime_table, phaseTT[ind]);
        if ((fp = fopen(fname, "r")) == NULL) {
            if (verbose)
                fprintf(errfp, "read_tt_tables: cannot open %s\n", fname);
            errorcode = 2;
            continue;
        }
/*
 *      number of distance and depth samples
 */
        fgets(buf, LINLEN, fp);
        skipcomments(buf, fp);
        sscanf(buf, "%d%d", &ndists, &ndepths);
        tt_tables[ind].ndel = ndists;
        tt_tables[ind].ndep = ndepths;
/*
 *      memory allocations
 */
        tt_tables[ind].deltas = (double *)calloc(ndists, sizeof(double));
        tt_tables[ind].depths = (double *)calloc(ndepths, sizeof(double));
        if (isdepthphase)
            tt_tables[ind].bpdel = alloc_matrix(ndists, ndepths);
        tt_tables[ind].tt = alloc_matrix(ndists, ndepths);
        tt_tables[ind].dtdd = alloc_matrix(ndists, ndepths);
        if ((tt_tables[ind].dtdh = alloc_matrix(ndists, ndepths)) == NULL) {
            free_tt_tbl(tt_tables);
            fclose(fp);
            errorcode = 1;
            return (TT_TABLE *) NULL;
        }
/*
 *      delta samples (broken into lines of 25 values)
 */
        m = ceil((double)ndists / 25.);
        for (i = 0, k = 0; k < m - 1; k++) {
            skipcomments(buf, fp);
            s = strtok(buf, " ");
            tt_tables[ind].deltas[i++] = atof(s);
            for (j = 1; j < 25; j++) {
                s = strtok(NULL, " ");
                tt_tables[ind].deltas[i++] = atof(s);
            }
        }
        if (i < ndists) {
            skipcomments(buf, fp);
            s = strtok(buf, " ");
            tt_tables[ind].deltas[i++] = atof(s);
            for (j = i; j < ndists; j++) {
                s = strtok(NULL, " ");
                tt_tables[ind].deltas[j] = atof(s);
            }
        }
/*
 *      depth samples
 */
        skipcomments(buf, fp);
        s = strtok(buf, " ");
        tt_tables[ind].depths[0] = atof(s);
        for (i = 1; i < ndepths; i++) {
            s = strtok(NULL, " ");
            tt_tables[ind].depths[i] = atof(s);
        }
/*
 *      travel-times (ndists rows, ndepths columns)
 */
        for (i = 0; i < ndists; i++) {
            skipcomments(buf, fp);
            s = strtok(buf, " ");
            tt_tables[ind].tt[i][0] = atof(s);
            for (j = 1; j < ndepths; j++) {
                s = strtok(NULL, " ");
                tt_tables[ind].tt[i][j] = atof(s);
            }
        }
/*
 *      dtdd (horizontal slowness)
 */
        for (i = 0; i < ndists; i++) {
            skipcomments(buf, fp);
            s = strtok(buf, " ");
            tt_tables[ind].dtdd[i][0] = atof(s);
            for (j = 1; j < ndepths; j++) {
                s = strtok(NULL, " ");
                tt_tables[ind].dtdd[i][j] = atof(s);
            }
        }
/*
 *      dtdh (vertical slowness)
 */
        for (i = 0; i < ndists; i++) {
            skipcomments(buf, fp);
            s = strtok(buf, " ");
            tt_tables[ind].dtdh[i][0] = atof(s);
            for (j = 1; j < ndepths; j++) {
                s = strtok(NULL, " ");
                tt_tables[ind].dtdh[i][j] = atof(s);
            }
        }
/*
 *      depth phase bounce point distances
 */
        if (isdepthphase) {
            for (i = 0; i < ndists; i++) {
                skipcomments(buf, fp);
                s = strtok(buf, " ");
                tt_tables[ind].bpdel[i][0] = atof(s);
                for (j = 1; j < ndepths; j++) {
                    s = strtok(NULL, " ");
                    tt_tables[ind].bpdel[i][j] = atof(s);
                }
            }
        }
        fclose(fp);
    }
    return tt_tables;
}

/*
 *  Title:
 *     free_tt_tbl
 *  Synopsis:
 *     Frees memory allocated to TT_TABLE structure
 *  Input Arguments:
 *     tt_tables - TT table structure
 *  Called by:
 *     read_data_files, read_tt_tables, main
 *  Calls:
 *     free_matrix
 */
void free_tt_tbl(TT_TABLE *tt_tables)
{
    int i, ndists = 0;
    for (i = 0; i < phaseTT_num; i++) {
        if ((ndists = tt_tables[i].ndel) == 0) continue;
        free_matrix(tt_tables[i].dtdh);
        free_matrix(tt_tables[i].dtdd);
        free_matrix(tt_tables[i].tt);
        if (phaseTT[i][0] == 'p' || phaseTT[i][0] == 's')
            free_matrix(tt_tables[i].bpdel);
        Free(tt_tables[i].depths);
        Free(tt_tables[i].deltas);
    }
    Free(tt_tables);
}

/*
 *  Title:
 *      read_etopo1
 *  Synopsis:
 *      Reads ETOPO1 topography file and store it in global short int topo
 *      array
 *      ETOPO1:
 *         etopo1_bed_g_i2.bin
 *         Amante, C. and B. W. Eakins,
 *           ETOPO1 1 Arc-Minute Global Relief Model: Procedures, Data Sources
 *           and Analysis.
 *           NOAA Technical Memorandum NESDIS NGDC-24, 19 pp, March 2009.
 *         NCOLS         21601
 *         NROWS         10801
 *         XLLCENTER     -180.000000
 *         YLLCENTER     -90.000000
 *         CELLSIZE      0.01666666667
 *         NODATA_VALUE  -32768
 *         BYTEORDER     LSBFIRST
 *         NUMBERTYPE    2_BYTE_INTEGER
 *         ZUNITS        METERS
 *         MIN_VALUE     -10898
 *         MAX_VALUE     8271
 *         1'x1' resolution, 21601 lons, 10801 lats
 *      Resampled ETOPO1 versions:
 *         etopo2_bed_g_i2.bin
 *           grdfilter -I2m etopo1_bed.grd -Fg10 -D4 -Getopo2_bed.grd
 *             Gridline node registration used
 *             x_min: -180 x_max: 180 x_inc: 0.0333333 name: nx: 10801
 *             y_min: -90 y_max: 90 y_inc: 0.0333333 name: ny: 5401
 *             z_min: -10648.7 z_max: 7399.13 name: m
 *             scale_factor: 1 add_offset: 0
 *         etopo5_bed_g_i2.bin
 *           grdfilter -I5m etopo1_bed.grd -Fg15 -D4 -Getopo5_bed.grd
 *             Gridline node registration used
 *             x_min: -180 x_max: 180 x_inc: 0.0833333 nx: 4321
 *             y_min: -90 y_max: 90 y_inc: 0.0833333 ny: 2161
 *             z_min: -10515.5 z_max: 6917.75 name: m
 *             scale_factor: 1 add_offset: 0
 *  ETOPO parameters are specified in config.txt file:
 *     etopofile - pathname for ETOPO file
 *     etoponlon - number of longitude samples in ETOPO
 *     etoponlat - number of latitude samples in ETOPO
 *     etopores  - cellsize in ETOPO
 *  Input Arguments:
 *     filename - filename pathname
 *  Return:
 *     topo - ETOPO bathymetry/elevation matrix or NULL on error
 *  Called by:
 *     read_data_files
 *  Calls:
 *     alloc_i2matrix, free_i2matrix
 */
short int **read_etopo1(char *filename)
{
    FILE *fp;
    short int **topo = (short int **)NULL;
    unsigned long n, m;
/*
 *  open etopo file
 */
    if ((fp = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Cannot open %s!\n", filename);
        errorcode = 2;
        return (short int **)NULL;
    }
/*
 *  allocate memory
 */
    if ((topo = alloc_i2matrix(etoponlat, etoponlon)) == NULL)
        return (short int **)NULL;
/*
 *  read etopo file
 */
    n = etoponlat * etoponlon;
    if ((m = fread(topo[0], sizeof(short int), n, fp)) != n) {
        fprintf(stderr, "Corrupted %s!\n", filename);
        fclose(fp);
        free_i2matrix(topo);
        return (short int **)NULL;
    }
    fclose(fp);
    return topo;
}

/*
 *  Title:
 *      find_etopo1
 *  Synopsis:
 *      Returns ETOPO1 topography in kilometers for a lat, lon pair.
 *  ETOPO parameters are specified in config.txt file:
 *     etopofile - pathname for ETOPO file
 *     etoponlon - number of longitude samples in ETOPO
 *     etoponlat - number of latitude samples in ETOPO
 *     etopores  - cellsize in ETOPO
 *  Input Arguments:
 *      lat, lon - latitude, longitude in degrees
 *      topo     - ETOPO bathymetry/elevation matrix
 *  Returns:
 *      elevation above sea level [km]
 *      topography above sea level is taken positive,
 *                 below sea level negative.
 *  Called by:
 *     topcor
 */
static double find_etopo1(double lat, double lon, short int **topo)
{
    int i, j, m, k1, k2;
    double a1, a2, lat2, lon2, lat1, lon1;
    double top, topo1, topo2, topo3, topo4;
/*
 *  bounding box
 */
    i = (int)((lon + 180.) / etopores);
    j = (int)((90. - lat) / etopores);
    lon1 = (double)(i) * etopores - 180.;
    lat1 = 90. - (double)(j) * etopores;
    lon2 = (double)(i + 1) * etopores - 180.;
    lat2 = 90. - (double)(j + 1) * etopores;
    k1 = i;
    k2 = i + 1;
    m = j;
    a1 = (lon2 - lon) / (lon2 - lon1);
    a2 = (lat2 - lat) / (lat2 - lat1);
/*
 *  take care of grid boundaries
 */
    if (i < 0 || i > etoponlon - 2) {
        k1 = etoponlon - 1;
        k2 = 0;
    }
    if (j < 0) {
        m = 0;
        a2 = 0.;
    }
    if (j > etoponlat - 2) {
        m = etoponlat - 2;
        a2 = 1.;
    }
/*
 *  interpolate
 */
    topo1 = (double)topo[m][k1];
    topo2 = (double)topo[m+1][k1];
    topo3 = (double)topo[m][k2];
    topo4 = (double)topo[m+1][k2];
    top = (1. - a1) * (1. - a2) * topo1 + a1 * (1. - a2) * topo3 +
          (1. - a1) * a2 * topo2 + a1 * a2 * topo4;
    return top / 1000.;
}

/*
 *  Title:
 *     get_phase_index
 *  Synopsis:
 *	   Returns index of tt_table struct array for a given phase
 *  Input Arguments:
 *     phase - phase
 *  Return:
 *     phase index or -1 on error
 *  Called by:
 *     read_ttime, depth_phase_stack
 */
int get_phase_index(char *phase)
{
    int i;
    for (i = 0; i < phaseTT_num; i++) {
        if (streq(phase, phaseTT[i])) return i;
    }
    return -1;
}

/*
 *  Title:
 *     get_tt
 *  Synopsis:
 *	   Returns TT table values for a given phase, depth and delta.
 *     Bicubic spline interpolation is used to get interpolated values.
 *     Horizontal and vertical slownesses are calculated if requested.
 *     Bounce point distance is calculated for depth phases.
 *  Input Arguments:
 *     tt_tablep - TT table structure for phase
 *     depth     - depth
 *     delta     - delta
 *     iszderiv  - do we need dtdh [0/1]?
 *  Output Arguments:
 *     dtdd  - interpolated dtdd (horizontal slowness, s/deg)
 *     dtdh  - interpolated dtdh (vertical slowness, s/km)
 *     bpdel - bounce point distance (deg) if depth phase
 *  Return:
 *     TT table value for a phase at depth and delta or -1. on error
 *  Called by:
 *     read_ttime
 *  Calls:
 *     bracket, spline, spline_int
 */
double get_tt(TT_TABLE *tt_tablep, double depth, double delta,
              int iszderiv, double *dtdd, double *dtdh, double *bpdel)
{
    int i, j, k, m, ilo, ihi, jlo, jhi, idel, jdep, ndep, ndel;
    int exactdelta = 0, exactdepth = 0, isdepthphase = 0;
    double ttim = -1., dydx = 0., d2ydx = 0.;
    double  x[DELTA_SAMPLES],  z[DEPTH_SAMPLES], d2y[DELTA_SAMPLES];
    double tx[DELTA_SAMPLES], tz[DEPTH_SAMPLES], tmp[DELTA_SAMPLES];
    double dx[DELTA_SAMPLES], dz[DEPTH_SAMPLES];
    double hx[DELTA_SAMPLES], hz[DEPTH_SAMPLES];
    double px[DELTA_SAMPLES], pz[DEPTH_SAMPLES];
    ndep = tt_tablep->ndep;
    ndel = tt_tablep->ndel;
/*
 *  for depth phases calculate bounce point distance
 */
    if (tt_tablep->phase[0] == 'p' || tt_tablep->phase[0] == 's')
        isdepthphase = 1;
    *bpdel = 0.;
    *dtdd  = -999.;
    *dtdh  = -999.;
/*
 *  check if travel time table exists
 */
    if (ndel == 0)
        return ttim;
/*
 *  check for out of range depth or delta
 */
    if (depth < tt_tablep->depths[0] || depth > tt_tablep->depths[ndep - 1] ||
        delta < tt_tablep->deltas[0] || delta > tt_tablep->deltas[ndel - 1]) {
        return ttim;
    }
/*
 *  delta range
 */
    bracket(delta, ndel, tt_tablep->deltas, &ilo, &ihi);
    if (fabs(delta - tt_tablep->deltas[ilo]) < DEPSILON) {
        idel = ilo;
        exactdelta = 1;
    }
    else if (fabs(delta - tt_tablep->deltas[ihi]) < DEPSILON) {
        idel = ihi;
        exactdelta = 1;
    }
    else if (ndel <= DELTA_SAMPLES) {
        ilo = 0;
        ihi = ndel;
        idel = ilo;
    }
    else {
        idel = ilo;
        ilo = idel - DELTA_SAMPLES / 2 + 1;
        ihi = idel + DELTA_SAMPLES / 2 + 1;
        if (ilo < 0) {
            ilo = 0;
            ihi = ilo + DELTA_SAMPLES;
        }
        if (ihi > ndel - 1) {
            ihi = ndel;
            ilo = ihi - DELTA_SAMPLES;
        }

    }
/*
 *  depth range
 */
    bracket(depth, ndep, tt_tablep->depths, &jlo, &jhi);
    if (fabs(depth - tt_tablep->depths[jlo]) < DEPSILON) {
        jdep = jlo;
        exactdepth = 1;
    }
    else if (fabs(depth - tt_tablep->depths[jhi]) < DEPSILON) {
        jdep = jhi;
        jlo = jhi;
        jhi++;
        exactdepth = 1;
    }
    else if (ndep <= DEPTH_SAMPLES) {
        jlo = 0;
        jhi = ndep;
        jdep = jlo;
    }
    else {
        jdep = jlo;
        jlo = jdep - DEPTH_SAMPLES / 2 + 1;
        jhi = jdep + DEPTH_SAMPLES / 2 + 1;
        if (jlo < 0) {
            jlo = 0;
            jhi = jlo + DEPTH_SAMPLES;
        }
        if (jhi > ndep - 1) {
            jhi = ndep;
            jlo = jhi - DEPTH_SAMPLES;
        }
    }
    if (exactdelta && exactdepth) {
        ttim  = tt_tablep->tt[idel][jdep];
        *dtdd  = tt_tablep->dtdd[idel][jdep];
        if (iszderiv)     *dtdh  = tt_tablep->dtdh[idel][jdep];
        if (isdepthphase) *bpdel = tt_tablep->bpdel[idel][jdep];
        return ttim;
    }
/*
 *  bicubic spline interpolation
 */
    for (k = 0, j = jlo; j < jhi; j++) {
/*
 *      no need for spline interpolation if exact delta
 */
        if (exactdelta) {
            if (tt_tablep->tt[idel][j] < 0) continue;
            z[k] = tt_tablep->depths[j];
            tz[k] = tt_tablep->tt[idel][j];
            if (isdepthphase) pz[k] = tt_tablep->bpdel[idel][j];
            dz[k] = tt_tablep->dtdd[idel][j];
            if (iszderiv)     hz[k] = tt_tablep->dtdh[idel][j];
            k++;
        }
/*
 *      spline interpolation in delta
 */
        else {
            for (m = 0, i = ilo; i < ihi; i++) {
                if (tt_tablep->tt[i][j] < 0) continue;
                x[m] = tt_tablep->deltas[i];
                tx[m] = tt_tablep->tt[i][j];
                if (isdepthphase) px[m] = tt_tablep->bpdel[i][j];
                dx[m] = tt_tablep->dtdd[i][j];
                if (iszderiv)     hx[m] = tt_tablep->dtdh[i][j];
                m++;
            }
            if (m < MIN_SAMPLES) continue;
            spline(m, x, tx, d2y, tmp);
            z[k] = tt_tablep->depths[j];
            tz[k] = spline_int(delta, m, x, tx, d2y, 0, &dydx, &d2ydx);
            if (isdepthphase) {
                spline(m, x, px, d2y, tmp);
                pz[k] = spline_int(delta, m, x, px, d2y, 0, &dydx, &d2ydx);
            }
            spline(m, x, dx, d2y, tmp);
            dz[k] = spline_int(delta, m, x, dx, d2y, 0, &dydx, &d2ydx);
            if (iszderiv) {
                spline(m, x, hx, d2y, tmp);
                hz[k] = spline_int(delta, m, x, hx, d2y, 0, &dydx, &d2ydx);
            }
            k++;
        }
    }
/*
 *  no valid data
 */
    if (k == 0) return ttim;
/*
 *  no need for spline interpolation if exact depth
 */
    if (exactdepth) {
        ttim = tz[0];
        if (isdepthphase) *bpdel = pz[0];
        *dtdd = dz[0];
        if (iszderiv)      *dtdh = hz[0];
        return ttim;
    }
/*
 *  insufficient data for spline interpolation
 */
    if (k < MIN_SAMPLES) return ttim;
/*
 *  spline interpolation in depth
 */
    spline(k, z, tz, d2y, tmp);
    ttim = spline_int(depth, k, z, tz, d2y, 0, &dydx, &d2ydx);
    if (isdepthphase) {
        spline(k, z, pz, d2y, tmp);
        *bpdel = spline_int(depth, k, z, pz, d2y, 0, &dydx, &d2ydx);
    }
    spline(k, z, dz, d2y, tmp);
    *dtdd = spline_int(depth, k, z, dz, d2y, 0, &dydx, &d2ydx);
    if (iszderiv) {
        spline(k, z, hz, d2y, tmp);
        *dtdh = spline_int(depth, k, z, hz, d2y, 0, &dydx, &d2ydx);
    }
    return ttim;
}


/*
 *  Title:
 *     calc_resid
 *  Synopsis:
 *     Calculates time residuals.
 *     Uses phase dependent information from <vmodel>_model.txt file.
 *        no_resid_phase - list of phases that don't get residuals, i.e.
 *                         never used in the location (e.g. amplitude phases)
 *     If mode is et to 'all' it attempts to get time residuals for all
 *        associated phases (final call),
 *     otherwise considers only time-defining phases.
 *  Input Arguments:
 *     sp        - pointer to current solution
 *     p         - array of phase structures
 *     mode      - "all" if require residuals for all phases
 *                 "use" if only want residuals for time-defining phases
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *     iszderiv  - calculate dtdh [0/1]?
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     event_loc, get_resids, fixed_hypo
 *  Calls:
 *     timeres
 */
int calc_resid(SOLREC *sp, PHAREC p[], char mode[4], EC_COEF *ec,
               TT_TABLE *tt_tables, short int **topo, int iszderiv)
{
    int i, all = 0;
/*
 *  set flag to enable/disable residual calculation for unused phases
 */
    if      (streq(mode, "all")) all = 1;
    else if (streq(mode, "use")) all = 0;
    else {
        fprintf(logfp,"calc_resid: unrecognised mode %s\n", mode);
        return 1;
    }
/*
 *  can't calculate residuals unless have a depth for the source
 */
    if (sp->depth == NULLVAL) {
        fprintf(logfp, "calc_resid: depthless hypocentre\n");
        return 1;
    }
/*
 *  won't get residuals if source has gone too deep
 */
    if (sp->depth > max_depth_km) {
        fprintf(logfp, "calc_resid: solution too deep %f > %f \n",
                sp->depth, max_depth_km);
        return 1;
    }
/*
 *  calculate time residual for associated/defining phases
 */
#ifdef WITH_GCD
/*
 *  use GCD (Mac OS) to parallelize the calculation of time residuals
 *  each phase is processed concurrently
 */
    dispatch_apply(sp->numphas, dispatch_get_global_queue(0, 0), ^(size_t i){
        p[i].resid = timeres(&p[i], all, sp, ec, tt_tables, topo, iszderiv);
    });
#else
/*
 *  single core
 */
    for (i = 0; i < sp->numphas; i++) {
        p[i].resid = timeres(&p[i], all, sp, ec, tt_tables, topo, iszderiv);
    }
#endif
    return 0;
}

/*
 *  Title:
 *     timeres
 *  Synopsis:
 *     Calculates the time residual for a single phase.
 *     Uses phase dependent information from <vmodel>_model.txt file.
 *        no_resid_phase - list of phases that don't get residuals, i.e.
 *                         never used in the location (e.g. amplitude phases)
 *  Input Arguments:
 *     pp        - pointer to phase struct
 *     all       - 1 if require residuals for all phases
 *                 0 if only want residuals for time-defining phases
 *     sp        - pointer to current solution
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *     iszderiv  - calculate dtdh [0/1]?
 *  Return:
 *     resid - time residual for a phase
 *  Called by:
 *     calc_resid
 *  Calls:
 *     read_ttime, reported_phase_resid
 */
static double timeres(PHAREC *pp, int all, SOLREC *sp, EC_COEF *ec,
                      TT_TABLE *tt_tables, short int **topo, int iszderiv)
{
    double obtime = 0., resid = NULLVAL;
    int j, isfirst = 0;
/*
 *  timeless or codeless phases don't have residuals
 */
    if (!all && (!pp->timedef || !pp->phase[0]))
        return resid;
    if (pp->time == NULLVAL)
        return resid;
    if (all) {
/*
 *      try to get residuals for all associated phases
 */
        if (pp->phase[0]) {
/*
 *          check for phases that don't get residuals (amplitudes etc)
 */
            for (j = 0; j < no_resid_phase_num; j++)
                if (streq(pp->phase, no_resid_phase[j]))
                    break;
            if (j != no_resid_phase_num)
                return resid;
        }
        else {
/*
 *          unidentified phase;
 *          try to get residual for the reported phase name
 */
            reported_phase_resid(sp, pp, ec, tt_tables, topo);
        }
    }
/*
 *  observed travel time
 */
    obtime = pp->time - sp->time;
#ifdef SERIAL
    if (verbose > 3)
        fprintf(logfp, "            %-6s %-8s obtime: %9.4f\n",
                pp->sta, pp->phase, obtime);
#endif
/*
 *  allow using first-arriving TT tables to deal with crossover distances?
 */
    if (all && !pp->timedef)
/*
 *      do not use first-arriving TT tables
 */
        isfirst = -1;
    else
/*
 *      allow
 */
        isfirst = 0;
/*
 *  predicted TT with corrections; partial derivatives if requested
 */
    if (read_ttime(sp, pp, ec, tt_tables, topo, iszderiv, isfirst)) {
/*
 *      no valid TT prediction
 */
        if (all && !pp->phase_fixed)
            strcpy(pp->phase, "");
        pp->ttime = resid = NULLVAL;
    }
    else
/*
 *      time residual
 */
        resid = obtime - pp->ttime;
#ifdef SERIAL
    if (verbose > 2) {
        fprintf(logfp, "        %-6s %-8s ", pp->sta, pp->phase);
        fprintf(logfp, "delta=%8.3f tt=", pp->delta);
        if (pp->ttime != NULLVAL) fprintf(logfp, "%9.4f ", pp->ttime);
        else                      fprintf(logfp, "%9s ", "");
        fprintf(logfp, "dtdd=%8.4f ", pp->dtdd);
        if (iszderiv) fprintf(logfp, "dtdh=%8.4f ", pp->dtdh);
        if (resid != NULLVAL) fprintf(logfp, "tres=%10.5f", resid);
        fprintf(logfp, "\n");
    }
#endif
    return resid;
}

/*
 *  Title:
 *     read_ttime
 *  Synopsis:
 *     Returns the travel-time prediction with elevation, ellipticity and
 *         optional bounce-point corrections for a phase.
 *     Horizontal and vertical slownesses are calculated if requested.
 *     The first two indices (0 and 1) in the TT table structures are
 *        reserved for the composite first-arriving P and S TT tables.
 *        The isfirst flags controls the use of these tables.
 *        1: in this case the phaseids are ignored and read_ttime returns
 *           the TT for the first-arriving P or S (never actually used);
 *        0: phaseids are respected, but first_arriving P or S tables can
 *           be used to get a valid TT table value at local/regional crossover
 *           distances without reidentifying the phase during the subsequent
 *           iterations of the location algorithm;
 *       -1: do not use them at all (the behaviour in phase id routines).
 *  Input Arguments:
 *     sp        - pointer to current solution.
 *     pp        - pointer to a phase record.
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *     iszderiv  - calculate dtdh [0/1]?
 *     isfirst   - use first arriving composite tables?
 *                    1: ignore phaseid and use them
 *                    0: don't use but allow for fix at crossover distances
 *                   -1: don't use (used in phase id routines)
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     calc_resid, reported_phase_resid, phase_id, same_time, id_pfake
 *  Calls:
 *     get_phase_index, get_tt, correct_ttime
 */
int read_ttime(SOLREC *sp, PHAREC *pp, EC_COEF *ec, TT_TABLE *tt_tables,
               short int **topo, int iszderiv, int isfirst)
{
    int pind = 0, isdepthphase = 0;
    double ttim = 0., dtdd = 0., dtdh = 0., bpdel = 0.;
/*
 *  invalid depth
 */
    if (sp->depth < 0. || sp->depth > max_depth_km || sp->depth == NULLVAL) {
        fprintf(logfp, "read_ttime: invalid depth! depth=%.2f phase=%s\n",
                sp->depth, pp->phase);
        return 1;
    }
/*
 *  invalid delta
 */
    if (pp->delta < 0. || pp->delta > 180. || pp->delta == NULLVAL) {
        fprintf(logfp, "read_ttime: invalid delta! delta=%.2f phase=%s\n",
                pp->delta, pp->phase);
        return 1;
    }
/*
 *  get travel-time table index for phase
 */
    if (isfirst == 1) {
/*
 *      use composite first-arriving P or S travel-time tables
 */
        if      (toupper(pp->phase[0]) == 'P') pind = 0;
        else if (toupper(pp->phase[0]) == 'S') pind = 1;
        else                                   pind = -1;
    }
    else
/*
 *      use travel-time tables specified for the phase
 */
        pind = get_phase_index(pp->phase);
    if (pind < 0) {
        if (verbose > 1) fprintf(logfp, "read_ttime: unknown phase %s!\n",
            pp->phase);
        return 1;
    }
/*
 *  if depth phase, we need dtdd for bounce point correction
 */
    isdepthphase = 0;
    if (pp->phase[0] == 'p' || pp->phase[0] == 's')
        isdepthphase = 1;
/*
 *  get travel-time table value
 */
    ttim = get_tt(&tt_tables[pind], sp->depth, pp->delta,
                  iszderiv, &dtdd, &dtdh, &bpdel);
/*
 *  couldn't get valid TT table value
 *     if we are allowed to use composite first-arriving travel-time tables,
 *     try them to deal with local/regional crossover ranges
 *     without renaming the phase
 */
    if (ttim < 0 && isfirst == 0) {
        if (pp->delta < 23) {
            if (streq(pp->phase, "Pg") || streq(pp->phase, "Pb") ||
                streq(pp->phase, "Pn") || streq(pp->phase, "P")) {
                ttim = get_tt(&tt_tables[0], sp->depth, pp->delta,
                              iszderiv, &dtdd, &dtdh, &bpdel);
            }
            if (streq(pp->phase, "Sg") || streq(pp->phase, "Sb") ||
                streq(pp->phase, "Sn") || streq(pp->phase, "S")) {
                ttim = get_tt(&tt_tables[1], sp->depth, pp->delta,
                              iszderiv, &dtdd, &dtdh, &bpdel);
            }
        }
    }
    if (ttim < 0) {
        if (verbose > 4) {
            fprintf(logfp, "        read_ttime: can't get TT for %s! ",
                    pp->phase);
            fprintf(logfp, "(depth=%.2f delta=%.2f)\n", sp->depth, pp->delta);
        }
        return 1;
    }
/*
 *  model predictions
 */
    pp->ttime = ttim;
    pp->dtdd = dtdd;
    if (iszderiv)      pp->dtdh = dtdh;
    if (isdepthphase) pp->bpdel = bpdel;
/*
 *  elevation, ellipticity and bounce point corrections
 */
    correct_ttime(sp, pp, ec, topo);
    return 0;
}

/*
 *  Title:
 *     correct_ttime
 *  Synopsis:
 *     Applies travel-time corrections to a predicted TT table value.
 *     Approximate geoid correction is calculated for Jeffreys-Bullen;
 *     otherwise the ak135 (Kennett and Gudmundsson, 1996) ellipticity
 *         correction is used.
 *     Bounce point correction is applied for depth phases, and for pwP,
 *        water depth correction is also calculated.
 *  Input Arguments:
 *     sp   - pointer to current solution.
 *     pp   - pointer to a phase structure.
 *     ec   - pointer to ellipticity correction coefficient structure
 *     topo - ETOPO bathymetry/elevation matrix
 *  Called by:
 *     read_ttime
 *  Calls:
 *     calc_geoid_corr (JB), get_ellip_corr (otherwise),
 *     get_elev_corr, get_bounce_corr
 */
static void correct_ttime(SOLREC *sp, PHAREC *pp, EC_COEF *ec,
                          short int **topo)
{
    double geoid_corr = 0., elev_corr = 0., ellip_corr = 0.;
    double bounce_corr = 0., water_corr = 0.;
    double f = (1. - FLATTENING) * (1. - FLATTENING);
    double ecolat = 0.;
/*
 *  keep approximate ellipticity corrections for JB (legacy)
 *  otherwise use ak135 ellipticity corrections (Kennett and Gudmundsson, 1996)
 */
    if (sp->lat != NULLVAL) {
/*
 *     apply travel time correction at station and epicentre for height
 *     of geoid above mean sphere. Only applies to JB.
 */
        if (streq(ttime_table, "jb")) {
            geoid_corr = calc_geoid_corr(sp->lat, pp);
            pp->ttime += geoid_corr;
#ifdef SERIAL
            if (verbose > 4)
                fprintf(logfp, "            %-6s geoid_corr=%.3f\n",
                        pp->sta,geoid_corr);
#endif
        }
/*
 *      ellipticity correction using Dziewonski and Gilbert (1976)
 *      formulation for ak135
 */
        else {
            ecolat = PI2 - atan(f * tan(DEG_TO_RAD * sp->lat));
            ellip_corr = get_ellip_corr(ec, pp->phase, ecolat, pp->delta,
                                        sp->depth, pp->esaz);
            pp->ttime += ellip_corr;
#ifdef SERIAL
            if (verbose > 4)
                fprintf(logfp, "            %-6s ellip_corr=%.3f\n",
                        pp->sta, ellip_corr);
#endif
        }
    }
/*
 *  elevation correction
 */
    elev_corr = get_elev_corr(pp);
    pp->ttime += elev_corr;
#ifdef SERIAL
    if (verbose > 4)
        fprintf(logfp, "            %-6s elev_corr=%.3f\n",
                pp->sta, elev_corr);
#endif
/*
 *  depth phase bounce point correction
 */
    if (pp->phase[0] == 'p' || pp->phase[0] == 's') {
        bounce_corr = get_bounce_corr(sp, pp, topo, &water_corr);
        pp->ttime += bounce_corr;
#ifdef SERIAL
        if (verbose > 4)
            fprintf(logfp, "            %-8s bounce_corr=%.3f\n",
                    pp->phase, bounce_corr);
#endif
/*
 *      water depth correction for pwP
 */
        if (streq(pp->phase, "pwP")) {
            pp->ttime += water_corr;
#ifdef SERIAL
            if (verbose > 4)
                fprintf(logfp, "            %-8s bounce_corr=%.3f\n",
                        pp->phase, water_corr);
#endif
        }
    }
}

/*
 *  Title:
 *     get_bounce_corr
 *  Synopsis:
 *     Returns the correction for topography/bathymetry at the bounce point
 *        for a depth phase, as well as the water depth correction for pwP.
 *     Adopted from Bob Engdahl's libtau extensions.
 *  Input Arguments:
 *     sp   - pointer to current solution.
 *     pp   - pointer to a phase structure.
 *     topo - ETOPO bathymetry/elevation matrix
 *  Output Arguments:
 *     tcorw - water travel time correction (water column)
 *  Return:
 *     tcorc - crust travel time correction (topography)
 *  Called by:
 *     correct_ttime
 *  Calls:
 *     deltaloc, topcor
 */
static double get_bounce_corr(SOLREC *sp, PHAREC *pp,
                              short int **topo, double *tcorw)
{
    int ips = 0;
    double tcor = 0., bp2 = 0., bpaz = 0., bplat = 0., bplon = 0.;
    *tcorw = 0.;
/*
 *  get geographic coordinates of bounce point
 */
    bp2 = pp->dtdd;
    bpaz = pp->esaz;
    if (bp2 < 0.)    bpaz += 180.;
    if (bpaz > 360.) bpaz -= 360.;
    deltaloc(sp->lat, sp->lon, pp->bpdel, bpaz, &bplat, &bplon);
/*
 *  get topography/bathymetry correction for upgoing part of depth phase
 */
    if      (strncmp(pp->phase, "pP", 2) == 0)  ips = 1;
    else if (strncmp(pp->phase, "pwP", 3) == 0) ips = 1;
    else if (strncmp(pp->phase, "pS", 2) == 0)  ips = 2;
    else if (strncmp(pp->phase, "sP", 2) == 0)  ips = 2;
    else if (strncmp(pp->phase, "sS", 2) == 0)  ips = 3;
    else                                        ips = 4;
    tcor = topcor(ips, bp2, bplat, bplon, topo, tcorw);
    return tcor;
}

/*
 *  Title:
 *     topcor
 *  Synopsis:
 *     Calculates bounce point correction for depth phases.
 *     Calculates water depth correction for pwP if water column > 1.5 km.
 *     Uses Bob Engdahl's topography equations.
 *  Input Arguments:
 *     ips   - 1 if pP* wave, 2 if sP* or pS* wave, 3 if sS* wave
 *     rayp  - horizontal slowness [s/deg]
 *     bplat - latitude of surface reflection point
 *     bplon - longitude of surface reflection point
 *     topo  - ETOPO bathymetry/elevation matrix
 *  Output Arguments:
 *     tcorw - water travel time correction (water column)
 *     wdep  - water column depth
 *  Return:
 *     tcorc - crust travel time correction (topography)
 *  Called by:
 *     get_bounce_corr
 *  Calls:
 *     find_etopo1
 */
double topcor(int ips, double rayp, double bplat, double bplon,
              short int **topo, double *tcorw)
{
    double watervel = 1.5;                    /* P velocity in water [km/s] */
    double delr = 0., term = 0., term1 = 0., term2 = 0.;
    double elev = 0., bp2 = 0., tcorc = 0.;
    *tcorw = 0.;
/*
 *  get topography/bathymetry elevation
 */
    elev = find_etopo1(bplat, bplon, topo);
    delr = elev;
    if (fabs(delr) < DEPSILON) return tcorc;
    bp2 = fabs(rayp) * RAD_TO_DEG / EARTH_RADIUS;
    if (ips == 1) {
/*
 *      pP*
 */
        term = psurfvel * psurfvel * bp2 * bp2;
        if (term > 1.) term = 1.;
        tcorc = 2. * delr * Sqrt(1. - term) / psurfvel;
        if (delr < -1.5) {
/*
 *          water depth is larger than 1.5 km
 */
            term = watervel * watervel * bp2 * bp2;
            if (term > 1.) term = 1.;
            *tcorw = -2. * delr * Sqrt(1. - term) / watervel;
        }
    }
    else if (ips == 2) {
/*
 *      pS* or sP*
 */
        term1 = psurfvel * psurfvel * bp2 * bp2;
        if (term1 > 1.) term1 = 1.;
        term2 = ssurfvel * ssurfvel * bp2 * bp2;
        if (term2 > 1.) term2 = 1.;
        tcorc = delr * (Sqrt(1. - term1) / psurfvel +
                        Sqrt(1. - term2) / ssurfvel);
    }
    else if (ips == 3) {
/*
 *      sS*
 */
        term = ssurfvel * ssurfvel * bp2 * bp2;
        if (term > 1.) term = 1.;
        tcorc = 2. * delr * Sqrt(1. - term) / ssurfvel;
    }
    else
        tcorc = 0.;
    return tcorc;
}

/*
 *  Title:
 *     get_elev_corr
 *  Synopsis:
 *     Calculates elevation correction for a station.
 *  Input Arguments:
 *     pp - pointer to phase structure.
 *  Return:
 *     Travel time correction for station elevation.
 *  Called by:
 *     correct_ttime
 *  Calls:
 *     lastlag
 */
static double get_elev_corr(PHAREC *pp)
{
    double elev_corr = 0., surfvel = 0.;
    int lastlag = 0;
/*
 *  unknown station elevation
 */
    if (pp->sta_elev == NULLVAL)
        return 0.;
/*
 *  find last lag of phase (P or S-type)
 */
    lastlag = last_lag(pp->phase);
    if (lastlag == 1)                 /* last lag is P */
        surfvel = psurfvel;
    else if (lastlag == 2)            /* last lag is S */
        surfvel = ssurfvel;
    else                             /* invalid/unknown */
        return 0.;
/*
 *  elevation correction
 */
    elev_corr  = surfvel * (pp->dtdd / DEG2KM);
    elev_corr *= elev_corr;
    if (elev_corr > 1.)
        elev_corr = 1./ elev_corr;
    elev_corr  = Sqrt(1. - elev_corr);
    elev_corr *= pp->sta_elev / (1000. * surfvel);
    return elev_corr;
}


/*
 *  Title:
 *     last_lag
 *  Synopsis:
 *     Finds last lag of phase (P or S-type)
 *  Input Arguments:
 *     phase - phase name
 *  Called by:
 *    get_elev_corr
 *  Return:
 *     1 if P-type, 2 if S-type, 0 otherwise
 */
static int last_lag(char phase[]) {
    int lastlag = 0;
    int i, n;
    n = (int)strlen(phase);
    if (n == 0)
        return lastlag;
    for (i = n - 1; i > -1; i--) {
        if (isupper(phase[i])) {
            if (phase[i] == 'P') {
                lastlag = 1;
                break;
            }
            if (phase[i] == 'S') {
                lastlag = 2;
                break;
            }
        }
    }
    return lastlag;
}


/*
 * Steps in delta used to get multiplier for geoid correction.
 */
int delta_step[] = { 9, 13, 22, 41, 63, 72, 83, 124, 140, 180 };

/*
 *  Title:
 *     calc_geoid_corr
 *  Synopsis:
 *     Calculates travel time correction for height above geoid.
 *     Approximate ellipticity correction for JB; legacy code.
 *  Input Arguments:
 *     lat - source latitude.
 *     pp  - pointer to  a phase structure.
 *  Return:
 *     Travel time correction for height of geoid.
 *  Called by:
 *     correct_ttime
 *  Calls:
 *     height_above_mean_sphere
 */
static double calc_geoid_corr(double lat, PHAREC *pp)
{
    double epi_hams = 0., sta_hams = 0.;
    double elcor = 0., geoid_corr = 0.;
    int i;
/*
 *  apply corrections at station and epicentre together
 *  delta_step array defined at top of file.
 */
    epi_hams = height_above_mean_sphere(lat * DEG_TO_RAD);
    sta_hams = height_above_mean_sphere(pp->sta_lat * DEG_TO_RAD);

    if (streq(pp->phase, "PKP")) {
        if (pp->delta < 140)
            elcor = 0.1;
        else
            elcor = 0.094;
    }
    else {
        for (i = 0; (int)(pp->delta + 0.5) > delta_step[i]; i++);
        elcor = 0.01 * i;
    }
    geoid_corr = elcor * (epi_hams + sta_hams);
    return geoid_corr;
}


/*
 *  Title:
 *     height_above_mean_sphere
 *  Synopsis:
 *     Calculates height above mean sphere (legacy code for JB).
 *  Input Arguments:
 *     lat  - latitude in radians.
 *  Called by:
 *     calc_geoid_corr
 *  Return:
 *     Height above mean sphere in km.
 */
static double height_above_mean_sphere(double lat)
{
    return 10.738 * cos(2. * lat) - 3.549 - 0.023 * cos(4. * lat);
}

/*  EOF  */

