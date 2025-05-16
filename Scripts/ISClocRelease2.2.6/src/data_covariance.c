#include "iscloc.h"
extern int verbose;
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;

/*
 * Functions:
 *    get_stalist
 *    starec_compare
 *    distance_matrix
 *    get_sta_index
 *    sort_phaserec_nn
 *    data_covariance_matrix
 *    read_variogram
 *    free_variogram
 */

/*
 *  Title:
 *     get_stalist
 *  Synopsis:
 *     Populates and sorts STAREC structure (stalist)with distinct stations.
 *  Input Arguments:
 *     numphas - number of associated phases
 *     p[]     - array of phase structures
 *  Ouput Arguments:
 *     nsta    - number of distinct stations
 *  Return:
 *     stalist[] - array of starec structures
 *  Called by:
 *     eventloc
 *  Calls:
 *     starec_compare
 */
STAREC *get_stalist(int numphas, PHAREC p[], int *nsta)
{
    STAREC *stalist = (STAREC *)NULL;
    char prev_prista[STALEN];
    int i, j, n = 0;
    strcpy(prev_prista, "");
/*
 *  count number of distinct stations
 */
    for (i = 0; i < numphas; i++) {
        if (streq(p[i].prista, prev_prista)) continue;
        strcpy(prev_prista, p[i].prista);
        n++;
    }
/*
 *  allocate memory for stalist
 */
    if ((stalist = (STAREC *)calloc(n, sizeof(STAREC))) == NULL) {
        fprintf(logfp, "get_stalist: cannot allocate memory\n");
        fprintf(errfp, "get_stalist: cannot allocate memory\n");
        errorcode = 1;
        return (STAREC *)NULL;
    }
/*
 *  populate stalist
 */
    strcpy(prev_prista, "");
    for (j = 0, i = 0; i < numphas; i++) {
        if (streq(p[i].prista, prev_prista)) continue;
        strcpy(prev_prista, p[i].prista);
        strcpy(stalist[j].fdsn, p[i].fdsn);
        strcpy(stalist[j].sta, p[i].sta);
        strcpy(stalist[j].altsta, p[i].prista);
        strcpy(stalist[j].agency, p[i].agency);
        strcpy(stalist[j].deploy, p[i].deploy);
        strcpy(stalist[j].lcn, p[i].lcn);
        stalist[j].lat = p[i].sta_lat;
        stalist[j].lon = p[i].sta_lon;
        stalist[j].elev = p[i].sta_elev;
        j++;
    }
    qsort(stalist, n, sizeof(STAREC), starec_compare);
    *nsta = n;
    return stalist;
}

/*
 *
 * starec_compare: compares two stalist records based on sta
 *
 */
int starec_compare(const void *sta1, const void *sta2)
{
    return strcmp(((STAREC *)sta1)->altsta, ((STAREC *)sta2)->altsta);
}

/*
 *  Title:
 *     distance_matrix
 *  Synopsis:
 *     Calculates station separations in km
 *  Input Arguments:
 *     nsta      - number of distinct stations
 *     stalist[] - array of starec structures
 *  Return:
 *     distmatrix
 *  Called by:
 *     eventloc
 *  Calls:
 *     alloc_matrix, distaz
 */
double **distance_matrix(int nsta, STAREC stalist[])
{
    double **distmatrix = (double **)NULL;
    int i, j, isnew = 0;
    double d = 0., esaz = 0., seaz = 0.;
/*
 *  memory allocation
 */
    if ((distmatrix = alloc_matrix(nsta, nsta)) == NULL) {
        fprintf(logfp, "station_separations: cannot allocate memory\n");
        fprintf(errfp, "station_separations: cannot allocate memory\n");
        errorcode = 1;
        return (double **)NULL;
    }
/*
 *  populate distmatrix; station separations in km
 */
    for (i = 0; i < nsta; i++) {
        isnew = 1;
        distmatrix[i][i] = 0.;
        for (j = i + 1; j < nsta; j++) {
            d = DEG2KM * distaz(stalist[j].lat, stalist[j].lon,
                                stalist[i].lat, stalist[i].lon,
	                            &seaz, &esaz, isnew);
            distmatrix[i][j] = distmatrix[j][i] = d;
            isnew = 0;
        }
    }
    if (verbose > 2) {
        fprintf(logfp, "    distance matrix (%d x %d):\n", nsta, nsta);
        for (i = 0; i < nsta; i++) {
            fprintf(logfp, "      %-6s ", stalist[i].sta);
            for (j = 0; j < nsta; j++)
                fprintf(logfp, "%7.1f ", distmatrix[i][j]);
            fprintf(logfp, "\n");
        }
    }
    return distmatrix;
}

/*
 *  Title:
 *     get_sta_index
 *  Synopsis:
 *     Returns index of a station in the stalist array
 *  Input Arguments:
 *     nsta      - number of distinct stations
 *     stalist[] - array of starec structures
 *     sta       - station to find
 *  Return:
 *     station index or -1 on error
 *  Called by:
 *     getndef, sort_phaserec_nn
 */
int get_sta_index(int nsta, STAREC stalist[], char *sta)
{
    int klo = 0, khi = nsta - 1, k = 0, i;
    if (nsta > 2) {
        while (khi - klo > 1) {
            k = (khi + klo) >> 1;
            if ((i = strcmp(stalist[k].altsta, sta)) == 0)
                return k;
            else if (i > 0)
                khi = k;
            else
                klo = k;
        }
        if (khi == 1) {
            k = 0;
            if (streq(sta, stalist[k].altsta)) return k;
            else return -1;
        }
        if (klo == nsta - 2) {
            k = nsta - 1;
            if (streq(sta, stalist[k].altsta)) return k;
            else return -1;
        }
    }
    else if (nsta == 2) {
        if (streq(sta, stalist[0].altsta)) return 0;
        else if (streq(sta, stalist[1].altsta)) return 1;
        else return -1;
    }
    else {
        if (streq(sta, stalist[0].altsta)) return 0;
        else return -1;
    }
    return -1;
}

/*
 *  Title:
 *     sort_phaserec_nn
 *  Synopsis:
 *     Sort phase structures so that they ordered by staorder, rdid, time.
 *     Ensures that phase records are ordered by the nearest-neighbour ordering,
 *     thus block-diagonalizing the data covariance matrix
 *  Input Arguments:
 *     numphas    - number of associated phases
 *     nsta       - number of distinct stations
 *     p[]        - array of phase structures.
 *     stalist[]  - array of starec structures
 *     staorder[] - array of staorder structures
 *  Called by:
 *     locate_event, na_search
 *  Calls:
 *     get_sta_index
 */
void sort_phaserec_nn(int numphas, int nsta, PHAREC p[],
                      STAREC stalist[], STAORDER staorder[])
{
    int i, j, k, kp, m;
    PHAREC temp;
/*
 *  sort by arrival time
 */
    for (i = 1; i < numphas; i++) {
        for (j = i - 1; j > -1; j--) {
            if ((p[j].time > p[j+1].time && p[j+1].time != NULLVAL) ||
                 p[j].time == NULLVAL) {
                swap(p[j], p[j+1]);
            }
        }
    }
/*
 *  sort by rdid
 */
    for (i = 1; i < numphas; i++) {
        for (j = i - 1; j > -1; j--) {
            if (p[j].rdid > p[j+1].rdid) {
                swap(p[j], p[j+1]);
            }
        }
    }
/*
 *  sort by nearest-neighbour station order
 */
    for (i = 1; i < numphas; i++) {
        for (j = i - 1; j > -1; j--) {
            m = get_sta_index(nsta, stalist, p[j].prista);
            kp = staorder[m].index;
            m = get_sta_index(nsta, stalist, p[j+1].prista);
            k = staorder[m].index;
            if (kp > k) {
                swap(p[j], p[j+1]);
            }
        }
    }
}

/*
 *  Title:
 *     data_covariance_matrix
 *  Synopsis:
 *     Constructs full data covariance matrix from variogram (model errors)
 *     and prior phase variances (measurement errors)
 *  Input Arguments:
 *     nsta       - number of distinct stations
 *     numphas    - number of associated phases
 *     nd         - number of defining phases
 *     p[]        - array of phase structures
 *     stalist[]  - array of starec structures
 *     distmatrix - matrix of station separations
 *     variogramp - pointer to generic variogram model
 *  Return:
 *     data covariance matrix
 *  Called by:
 *     locate_event, na_search
 *  Calls:
 *     alloc_matrix, free_matrix, get_sta_index, spline_int
 */
double **data_covariance_matrix(int nsta, int numphas, int nd, PHAREC p[],
                                STAREC stalist[], double **distmatrix,
                                VARIOGRAM *variogramp)
{
    int i, j, k, m, sind1 = 0, sind2 = 0;
    double stasep = 0., var = 0., dydx = 0., d2ydx = 0.;
    double **dcov = (double **)NULL;
/*
 *  allocate memory for dcov
 */
    if ((dcov = alloc_matrix(nd, nd)) == NULL) {
        fprintf(logfp, "data_covariance_matrix: cannot allocate memory\n");
        fprintf(errfp, "data_covariance_matrix: cannot allocate memory\n");
        errorcode = 1;
        return (double **)NULL;
    }
/*
 *  construct data covariance matrix from variogram and prior measurement
 *  error variances
 */
    for (k = 0, i = 0; i < numphas; i++) {
        if (!p[i].timedef) continue;
        if ((sind1 = get_sta_index(nsta, stalist, p[i].prista)) < 0) {
            free_matrix(dcov);
            return (double **)NULL;
        }
/*
 *      prior picking error variances add to the diagonal
 */
        dcov[k][k] = variogramp->sill + p[i].measerr * p[i].measerr;
        p[i].covindex = k;
        if (verbose > 4) {
            fprintf(logfp, "                i=%d k=%d sind1=%d ", i, k, sind1);
            fprintf(logfp, "sta=%s phase=%s\n", p[i].prista, p[i].phase);
        }
/*
 *      covariances
 */
        for (m = k + 1, j = i + 1; j < numphas; j++) {
            if (!p[j].timedef) continue;
/*
 *          different phases have different ray paths so they do not correlate
 */
            if (strcmp(p[i].phase, p[j].phase)) {
                m++;
                continue;
            }
            if ((sind2 = get_sta_index(nsta, stalist, p[j].prista)) < 0) {
                free_matrix(dcov);
                return (double **)NULL;
            }
/*
 *          station separation
 */
            var = 0.;
            stasep = distmatrix[sind1][sind2];
            if (stasep < variogramp->maxsep) {
/*
 *              interpolate variogram
 */
                var = spline_int(stasep, variogramp->n, variogramp->x,
                                 variogramp->y, variogramp->d2y,
                                 0, &dydx, &d2ydx);
/*
 *              covariance: sill - variogram
 */
                var = variogramp->sill - var;
            }
            dcov[k][m] = var;
            dcov[m][k] = var;
            if (verbose > 4) {
                fprintf(logfp, "                  j=%d m=%d sind2=%d ",
                        j, m, sind2);
                fprintf(logfp, "sta=%s phase=%s ", p[j].prista, p[j].phase);
                fprintf(logfp, "stasep=%.1f var=%.3f\n", stasep, var);
            }
            m++;
        }
        k++;
    }
    if (verbose > 2) {
        fprintf(logfp, "        Data covariance matrix C(%d x %d):\n", nd, nd);
        for (i = 0, k = 0; k < numphas; k++) {
            if (!p[k].timedef) continue;
            fprintf(logfp, "          %4d %-6s %-8s ",
                    i, p[k].prista, p[k].phase);
            for (j = 0; j < nd; j++) fprintf(logfp, "%6.4f ", dcov[i][j]);
            fprintf(logfp, "\n");
            i++;
        }
    }
    return dcov;
}

/*
 *  Title:
 *     read_variogram
 *  Synopsis:
 *     Reads generic variogram from file and stores it in VARIOGRAM structure.
 *  Input Arguments:
 *     fname - pathname of variogram file
 *  Output Arguments:
 *     variogramp - pointer to VARIOGRAM structure
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     read_data_files
 *  Calls:
 *     spline, Free, skipcomments
 */
int read_variogram(char *fname, VARIOGRAM *variogramp)
{
    FILE *fp;
    char buf[LINLEN];
    int i, n = 0;
    double sill = 0., maxsep = 0.;
    double *tmp = (double *)NULL;
    char *s;
/*
 *  open variogram file and get number of phases
 */
    if ((fp = fopen(fname, "r")) == NULL) {
        fprintf(logfp, "read_variogram: cannot open %s\n", fname);
        fprintf(errfp, "read_variogram: cannot open %s\n", fname);
        errorcode = 2;
        return 1;
    }
/*
 *  number of samples
 */
    fgets(buf, LINLEN, fp);
    skipcomments(buf, fp);
    sscanf(buf, "%d", &n);
    variogramp->n = n;
/*
 *  sill variance
 */
    skipcomments(buf, fp);
    sscanf(buf, "%lf", &sill);
    variogramp->sill = sill;
/*
 *  max station separation to be considered
 */
    skipcomments(buf, fp);
    sscanf(buf, "%lf", &maxsep);
    variogramp->maxsep = maxsep;
/*
 *  memory allocations
 */
    tmp = (double *)calloc(n, sizeof(double));
    variogramp->x = (double *)calloc(n, sizeof(double));
    variogramp->y = (double *)calloc(n, sizeof(double));
    if ((variogramp->d2y = (double *)calloc(n, sizeof(double))) == NULL) {
        fprintf(logfp, "read_variogram: cannot allocate memory!\n");
        fprintf(errfp, "read_variogram: cannot allocate memory!\n");
        Free(variogramp->y);
        Free(variogramp->x);
        Free(tmp);
        errorcode = 1;
        return 1;
    }
/*
 *  variogram: x = distance [km], y = gamma(x) [s**2]
 */
    for (i = 0; i < n; i++) {
       skipcomments(buf, fp);
       s = strtok(buf, " ");
       variogramp->x[i] = atof(s);
       s = strtok(NULL, " ");
       variogramp->y[i] = atof(s);
    }
    fclose(fp);
/*
 *  second derivatives of the natural spline interpolating function
 */
    spline(n, variogramp->x, variogramp->y, variogramp->d2y, tmp);
    Free(tmp);
    return 0;
}

/*
 *  Title:
 *     free_variogram
 *  Synopsis:
 *     frees memory allocated to VARIOGRAM structure
 *  Input Arguments:
 *     variogramp - pointer to VARIOGRAM structure
 *  Called by:
 *     main, read_data_files
 *  Calls:
 *     Free
 */
void free_variogram(VARIOGRAM *variogramp)
{
    Free(variogramp->d2y);
    Free(variogramp->y);
    Free(variogramp->x);
}

