#include "iscloc.h"
extern int verbose;   /* verbose level */
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;
extern int numagencies;
extern char agencies[MAXBUF][AGLEN];

/*
 * Functions:
 *    calc_netmag
 *    read_magQ
 */

/*
 * Local functions
 *    netmag
 *    calc_sta_mb
 *    calc_sta_ms
 *    get_magQ
 *    stamag_compare
 *    rdmag_compare
 *    rdmagsta_compare
 */
static int netmag(SOLREC *sp, READING *rdindx, PHAREC p[], STAMAG *stamag,
                  RDMAG *rdmag, MSZH *mszh, int ismb, int isQ, MAGQ *magQ);
static double calc_sta_mb(int start, int n, PHAREC p[], int mtypeid,
                          double depth, int isQ, MAGQ *magQ);
static double calc_sta_ms(int start, int n, PHAREC p[], int mtypeid,
                          MSZH *mszh);
static double get_magQ(double delta, double depth, int isQ, MAGQ *magQ);
static int stamag_compare(const void *smag1, const void *smag2);
static int rdmag_sort(int nass, RDMAG *rdmag);

/*
 *  Title:
 *     calc_netmag
 *  Synopsis:
 *     Calculates network Ms and mb.
 *  Input Arguments:
 *     sp        - pointer to current solution
 *     rdindx    - array of reading structures
 *     p[]       - array of phase structures
 *     stamag_mb - array of station mb structures
 *     stamag_ms - array of station MS structures
 *     rdmag_mb  - array of reading mb structures
 *     rdmag_ms  - array of reading MS structures
 *     mszh      - array of MS vertical/horizontal magnitude structures
 *     ismbQ     - use mb magnitude attenuation table? (0/1)
 *     mbQ       - pointer to mb Q(d,h) table
 *  Return:
 *     0/1 on success/error.
 *  Called by:
 *     eventloc
 *  Calls:
 *     netmag
 */
int calc_netmag(SOLREC *sp, READING *rdindx, PHAREC p[],
                STAMAG *stamag_mb, STAMAG *stamag_ms,
                RDMAG *rdmag_mb, RDMAG *rdmag_ms, MSZH *mszh,
                int ismbQ, MAGQ *mbQ)
{
    extern double surf_mag_max_depth;                /* From config file */
    sp->bodymag = NULLVAL;
    sp->bodymag_uncertainty = 0.;
    sp->nsta_mb = 0;
    sp->nass_mb = 0;
    sp->nmbagency = 0;
    sp->surfmag = NULLVAL;
    sp->surfmag_uncertainty = 0.;
    sp->nsta_ms = 0;
    sp->nass_ms = 0;
    sp->nMsagency = 0;
/*
 *  network mb
 */
    if (netmag(sp, rdindx, p, stamag_mb, rdmag_mb, mszh, 1, ismbQ, mbQ))
        return 1;
    if (verbose) {
        if (sp->bodymag != NULLVAL)
            fprintf(logfp, "    bodymag=%.2f stderr=%.2f nsta=%d\n",
                    sp->bodymag, sp->bodymag_uncertainty, sp->nsta_mb);
        else
            fprintf(logfp, "    No data for mb\n");
    }
/*
 *  Ms - only do if shallow enough.
 */
    if (sp->depth > surf_mag_max_depth)
        return 0;
/*
 *  network Ms
 */
    if (netmag(sp, rdindx, p, stamag_ms, rdmag_ms, mszh, 0, 0, (MAGQ *)NULL))
        return 1;
    if (verbose) {
        if (sp->surfmag != NULLVAL)
            fprintf(logfp, "    surfmag=%.2f stderr=%.2f nsta=%d\n",
                    sp->surfmag, sp->surfmag_uncertainty, sp->nsta_ms);
        else
            fprintf(logfp, "    No data for Ms\n");
    }
    return 0;
}

/*
 *  Title:
 *     netmag
 *  Synopsis:
 *     Calculates network magnitude.
 *     At least three station magnitudes are required for a network magnitude.
 *     The network magnitude (mb or Ms) is defined as the median of the
 *         station magnitudes.
 *     The network magnitude uncertainty is defined as the SMAD of the
 *         alpha-trimmed station magnitudes.
 *     The station magnitude is defined as the median of reading magnitudes
 *         for a station.
 *     The reading magnitude is defined as the magnitude computed from the
 *         maximal log(A/T) in a reading.
 *  Input Arguments:
 *     sp     - pointer to current solution
 *     rdindx - array of reading structures
 *     p[]    - array of phase structures
 *     stamag - array of station magnitude structures
 *     rdmag  - array of reading magnitude structures
 *     mszh   - array of MS vertical/horizontal magnitude structures
 *     ismb   - 1 if mb, 0 otherwise
 *     isQ    - use magnitude attenuation table? (0/1)
 *     magQ   - pointer to MAGQ Q(d,h) table
 *     depth  - source depth
 *  Return:
 *     0/1 on success/error.
 *  Called by:
 *     calc_netmag
 *  Calls:
 *     calc_sta_mb, calc_sta_ms, stamag_compare, Free
 */
static int netmag(SOLREC *sp, READING *rdindx, PHAREC p[], STAMAG *stamag,
                  RDMAG *rdmag, MSZH *mszh, int ismb, int isQ, MAGQ *magQ)
{
    extern double mag_range_warn_thresh;             /* From config file */
    char prev_prista[STALEN], mtype[4];
    double *adev = (double *)NULL;
    double reading_mag = 0., median = 0., smad = 0.;
    double min_mag = 0., max_mag = 0.;
    int nass = 0, nsta = 0, nagent = 0, mtypeid = 0;
    int ageindex[MAXBUF];
    int i, j, m, np;
/*
 *  Set magnitude type
 */
    if (ismb) {
        strcpy(mtype, "mb");
        mtypeid = 1;
    }
    else {
        strcpy(mtype, "MS");
        mtypeid = 2;
    }
/*
 *  loop over reading
 */
    for (i = 0; i < sp->nreading; i++) {
        m = rdindx[i].start;
        np = rdindx[i].start + rdindx[i].npha;
/*
 *      get reading magnitude
 */
        if (ismb)
            reading_mag = calc_sta_mb(m, np, p, mtypeid, sp->depth, isQ, magQ);
        else
            reading_mag = calc_sta_ms(m, np, p, mtypeid, &mszh[nass]);
/*
 *      populate rdmag record
 */
        if (reading_mag > -999.) {
            rdmag[nass].rdid = p[m].rdid;
            rdmag[nass].magdef = 0;
            rdmag[nass].mtypeid = mtypeid;
            strcpy(rdmag[nass].sta, p[m].prista);
            strcpy(rdmag[nass].agency, p[m].agency);
            rdmag[nass].magnitude = reading_mag;
            nass++;
        }
    }
/*
 *  no amplitudes were reported
 */
    if (!nass)
        return 0;
/*
 *  number of readings contributing to netmag
 */
    if (ismb) sp->nass_mb = nass;
    else      sp->nass_ms = nass;
/*
 *  agencies
 */
    for (i = 0; i < numagencies; i++) ageindex[i] = 0;
    for (j = 0; j < nass; j++) {
        for (i = 0; i < numagencies; i++)
            if (streq(rdmag[j].agency, agencies[i]))
                ageindex[i]++;
    }
/*
 *  sort reading magnitudes by station and magnitude
 */
    rdmag_sort(nass, rdmag);
/*
 *  define stamag as median of reading magnitudes for the same station
 */
    strcpy(prev_prista, rdmag[0].sta);
    j = 0;
    nsta = 0;
    for (i = 0; i < nass; i++) {
        if (strcmp(rdmag[i].sta, prev_prista) && j) {
            m = j / 2;
            if (j % 2) {
                median = rdmag[i-m-1].magnitude;
                rdmag[i-m-1].magdef = 1;
            }
            else {
                median = 0.5 * (rdmag[i-m-1].magnitude + rdmag[i-m].magnitude);
                rdmag[i-m-1].magdef = 1;
                rdmag[i-m].magdef = 1;
            }
            j = 0;
/*
 *          populate stamag record for previous sta
 */
            strcpy(stamag[nsta].sta, prev_prista);
            strcpy(stamag[nsta].agency, rdmag[i-m-1].agency);
            stamag[nsta].magnitude = median;
            stamag[nsta].magdef = 0;
            stamag[nsta].mtypeid = mtypeid;
            nsta++;
        }
        strcpy(prev_prista, rdmag[i].sta);
        j++;
    }
/*
 *  last sta
 */
    if (j) {
        m = j / 2;
        if (j % 2) {
            median = rdmag[i-m-1].magnitude;
            rdmag[i-m-1].magdef = 1;
        }
        else {
            median = 0.5 * (rdmag[i-m-1].magnitude + rdmag[i-m].magnitude);
            rdmag[i-m-1].magdef = 1;
            rdmag[i-m].magdef = 1;
        }
        strcpy(stamag[nsta].sta, prev_prista);
        strcpy(stamag[nsta].agency, rdmag[i-m-1].agency);
        stamag[nsta].magnitude = median;
        stamag[nsta].magdef = 0;
        stamag[nsta].mtypeid = mtypeid;
        nsta++;
    }
/*
 *  number of station magnitudes contributing to netmag
 */
    if (ismb) sp->nsta_mb = nsta;
    else      sp->nsta_ms = nsta;
/*
 *  insufficient number of station magnitudes?
 */
    if (nsta < 3) {
        fprintf(logfp, "    netmag: %s: insufficient number of stations %d\n",
                mtype, nsta);
        return 0;
    }
/*
 *  network magnitude is calculated as an alpha-trimmed median; alpha = 20%
 */
    qsort(stamag, nsta, sizeof(STAMAG), stamag_compare);
    i = nsta / 2;
    if (nsta % 2) median = stamag[i].magnitude;
    else          median = 0.5 * (stamag[i - 1].magnitude +
                                  stamag[i].magnitude);
/*
 *  alpha trim range
 */
    m = floor(0.2 * (double)nsta);
    min_mag = stamag[m].magnitude;
    max_mag = stamag[nsta - m - 1].magnitude;
/*
 *  allocate memory for adev
 */
    if ((adev = (double *)calloc(nsta, sizeof(double))) == NULL) {
        fprintf(logfp, "netmag: cannot allocate memory\n");
        fprintf(errfp, "netmag: cannot allocate memory\n");
        errorcode = 1;
        return 0;
    }
/*
 *  netmag uncertainty: smad of alpha-trimmed station magnitudes
 */
    for (j = 0, i = m; i < nsta - m; j++, i++) {
        adev[j] = fabs(stamag[i].magnitude - median);
        stamag[i].magdef = 1;
    }
    qsort(adev, j, sizeof(double), double_compare);
    m = j / 2;
    if (j % 2) smad = 1.4826 * adev[m];
    else       smad = 1.4826 * 0.5 * (adev[m - 1] + adev[m]);
    Free(adev);
/*
 *  report station magnitudes
 */
    fprintf(logfp, "    %s station magnitudes\n", mtype);
    for (i = 0; i < nsta; i++)
        fprintf(logfp, "      %-6s stamag=%.3f magdef=%d\n",
                stamag[i].sta, stamag[i].magnitude, stamag[i].magdef);
/*
 *  number of reporting agencies
 */
    fprintf(logfp, "    agencies contributing to ISC %s\n", mtype);
    for (nagent = 0, i = 0; i < numagencies; i++) {
        if (ageindex[i]) {
            nagent++;
            fprintf(logfp, "      %3d %-17s %4d readings\n",
                    nagent, agencies[i], ageindex[i]);
        }
    }
/*
 *  network magnitude
 */
    if (ismb) {
        sp->bodymag = median;
        sp->bodymag_uncertainty = smad;
        sp->nmbagency = nagent;
    }
    else {
        sp->surfmag = median;
        sp->surfmag_uncertainty = smad;
        sp->nMsagency = nagent;
    }
/*
 *  Test for big differences in station magnitudes.
 */
    if ((max_mag - min_mag) > mag_range_warn_thresh)
        fprintf(logfp, "WARNING: %s RANGE %.1f - %.1f\n",
                mtype, min_mag, max_mag);
    return 0;
}

/*
 *
 * stamag_compare: compares two stamag records based on the magnitude
 *
 */
static int stamag_compare(const void *smag1, const void *smag2)
{
    if (((STAMAG *)smag1)->magnitude < ((STAMAG *)smag2)->magnitude)
        return -1;
    if (((STAMAG *)smag1)->magnitude > ((STAMAG *)smag2)->magnitude)
        return 1;
    return 0;
}

/*
 *
 * rdmag_sort: sorts rdmag records by station and magnitude
 *
 */
static int rdmag_sort(int nass, RDMAG *rdmag)
{
    int i, j;
    RDMAG temp;
    for (i = 1; i < nass; i++) {
        for (j = i - 1; j > -1; j--) {
            if (rdmag[j].magnitude > rdmag[j+1].magnitude) {
                swap(rdmag[j], rdmag[j+1]);
            }
        }
    }
    for (i = 1; i < nass; i++) {
        for (j = i - 1; j > -1; j--) {
            if (strcmp(rdmag[j].sta, rdmag[j+1].sta) > 0) {
                swap(rdmag[j], rdmag[j+1]);
            }
        }
    }
    return 0;
}

/*
 *  Title:
 *     calc_sta_mb
 *  Synopsis:
 *     Calculates mb for a single reading.
 *     Finds the reported amplitude, period pair for which A/T is maximal.
 *         Amplitude mb = log(A/T) + Q(d,h)
 *         Reading   mb = log(max(A/T)) + Q(d,h)
 *     If no amplitude, period pairs are reported, use the reported logat values
 *         Amplitude mb = logat + Q(d,h)
 *         Reading   mb = max(logat) + Q(d,h)
 *  Input Arguments:
 *     start - starting index in a reading
 *     n     - number of phases in the reading
 *     p[]   - array of phase structures
 *     depth - source depth
 *     isQ   - use magnitude attenuation table? (0/1/2)
 *               0 - do not use
 *               1 - Gutenberg-Richter
 *               2 - Veith-Clawson
 *     magQ  - pointer to MAGQ Q(d,h) table
 *  Return:
 *     reading_mb - mb for this reading
 *  Called by:
 *     netmag
 *  Calls:
 *     get_magQ
 */
static double calc_sta_mb(int start, int n, PHAREC p[], int mtypeid,
                          double depth, int isQ, MAGQ *magQ)
{
    extern double body_mag_min_dist;            /* From config file */
    extern double body_mag_max_dist;            /* From config file */
    extern double body_mag_min_per;             /* From config file */
    extern double body_mag_max_per;             /* From config file */
    extern char mb_phase[MAXNUMPHA][PHALEN];  /* From model file  */
    extern int mb_phase_num;                    /* From model file  */
    double logat = 0., amp = 0., apert = 0., delta = 0.;
    double reading_mb = -999., q = 0., maxat = 0.;
    int i, j, ind_amp = 0, ind_pha = 0;
/*
 *
 *  calculate reading mb from max(A/T) in a reading
 *
 */
    ind_amp = ind_pha = -1;
    maxat = -999.;
    delta = p[start].delta;
    for (i = start; i < n; i++) {
/*
 *      Only need look at phases with amplitudes
 */
        if (p[i].numamps == 0)
            continue;
        if (p[i].purged)       /* used by iscloc_search */
            continue;
/*
 *      Only consider phases contributing to mb
 */
        for (j = 0; j < mb_phase_num; j++)
            if (streq(p[i].phase, mb_phase[j]))
                break;
        if (j == mb_phase_num)
            continue;
/*
 *      Ignore readings recorded outside the mb distance range
 */
        if (delta < body_mag_min_dist || delta > body_mag_max_dist)
            continue;
/*
 *      Loop over amplitudes
 */
        for (j = 0; j < p[i].numamps; j++) {
/*
 *          ignore amplitudes measured on horizontal components
 */
            if (p[i].a[j].comp == 'N' || p[i].a[j].comp == 'E')
                continue;
/*
 *          ignore amplitudes outside the period range
 */
            if (p[i].a[j].per < body_mag_min_per ||
                p[i].a[j].per > body_mag_max_per)
                continue;
/*
 *          ignore null amplitudes
 */
            if (fabs(p[i].a[j].amp) < DEPSILON || p[i].a[j].amp == NULLVAL)
                continue;
/*
 *          ignore null periods
 */
            if (fabs(p[i].a[j].per) < DEPSILON || p[i].a[j].per == NULLVAL)
                continue;
/*
 *          for VC corrections amplitudes are measured as peak-to-peak
 */
            amp = (isQ == 2) ? 2. * p[i].a[j].amp : p[i].a[j].amp;
            apert = amp / p[i].a[j].per;
/*
 *          keep track of maximum amplitude
 */
            if (apert > maxat) {
                maxat = apert;
                ind_amp = j;
                ind_pha = i;
            }
/*
 *          amplitude magnitude
 *          mb = log(A/T) + Q(d,h)
 */
            logat = log10(apert);
            q = get_magQ(delta, depth, isQ, magQ);
            p[i].a[j].magnitude = logat + q;
            p[i].a[j].ampdef = 0;
            p[i].a[j].mtypeid = mtypeid;
            strcpy(p[i].a[j].magtype, "mb");
            if (verbose > 2) {
                fprintf(logfp, "          i=%2d j=%2d amp=%.2f ",
                        i, j, p[i].a[j].amp);
                fprintf(logfp, "per=%.2f logat%.2f magnitude: %.2f\n",
                        p[i].a[j].per, logat, p[i].a[j].magnitude);
            }
        }
    }
/*
 *  take the magnitude belonging to the maximal A/T as the
 *  reading magnitude
 *  mb = log(max(A/T)) + Q(d,h)
 */
    if (ind_pha > -1) {
        p[ind_pha].a[ind_amp].ampdef = 1;
        reading_mb = p[ind_pha].a[ind_amp].magnitude;
        if (verbose > 1)
            fprintf(logfp, "        rdid=%d %-6s depth=%.2f delta=%.2f mb=%.2f\n",
                    p[ind_pha].rdid, p[ind_pha].sta, depth, delta, reading_mb);
        return reading_mb;
    }
/*
 *
 *  if no amp/per pairs were reported, calculate reading mb from
 *      the maximal reported logat value
 *
 */
    ind_amp = ind_pha = -1;
    maxat = -999;
    for (i = start; i < n; i++) {
/*
 *      Only need look at phases with amplitudes.
 */
        if (p[i].numamps == 0)
            continue;
        if (p[i].purged)       /* used by iscloc_search */
            continue;
/*
 *      Only consider phases contributing to mb
 */
        for (j = 0; j < mb_phase_num; j++)
            if (streq(p[i].phase, mb_phase[j]))
                break;
        if (j == mb_phase_num)
            continue;
/*
 *      Ignore readings recorded outside the mb distance range
 */
        if (delta < body_mag_min_dist || delta > body_mag_max_dist)
            continue;
/*
 *      Loop over amplitudes
 */
        for (j = 0; j < p[i].numamps; j++) {
            if (p[i].a[j].logat != NULLVAL ) {
                logat = p[i].a[j].logat;
/*
 *              keep track of maximum logat
 */
                if (logat > maxat) {
                    maxat = logat;
                    ind_amp = j;
                    ind_pha = i;
                }
/*
 *              amplitude magnitude
 *              mb = logat + Q(d,h)
 */
                q = get_magQ(delta, depth, isQ, magQ);
                p[i].a[j].magnitude = logat + q;
                p[i].a[j].ampdef = 0;
                p[i].a[j].mtypeid = mtypeid;
                strcpy(p[i].a[j].magtype, "mb");
                if (verbose > 2) {
                    fprintf(logfp, "          i=%2d j=%2d amp=%.2f ",
                        i, j, p[i].a[j].amp);
                    fprintf(logfp, "per=%.2f logat%.2f magnitude: %.2f\n",
                        p[i].a[j].per, logat, p[i].a[j].magnitude);
                }
            }
        }
    }
/*
 *  take the magnitude belonging to the maximal logat as the
 *  reading mb
 *  mb = max(logat) + Q(d,h)
 */
    if (ind_pha > -1) {
        p[ind_pha].a[ind_amp].ampdef = 1;
        reading_mb = p[ind_pha].a[ind_amp].magnitude;
        if (verbose > 1)
            fprintf(logfp, "        rdid=%d %-6s depth=%.2f delta=%.2f mb=%.2f\n",
                    p[ind_pha].rdid, p[ind_pha].sta, depth, delta, reading_mb);
        return reading_mb;
    }
    return -999.;
}

/*
 *  Title:
 *     get_magQ
 *  Synopsis:
 *     Calculates magnitude attenuation Q(d,h) value.
 *     Gutenberg, B. and C.F. Richter, 1956,
 *         Magnitude and energy of earthquakes,
 *         Ann. Geof., 9, 1-5.
 *     Veith, K.F. and G.E. Clawson, 1972,
 *         Magnitude from short-period P-wave data,
 *         Bull. Seism. Soc. Am., 62, 2, 435-452.
 *     Murphy, J.R. and B.W. Barker, 2003,
 *         Revised B(d,h) correction factors for use
 *         in estimation of mb magnitudes,
 *         Bull. Seism. Soc. Am., 93, 1746-1764.
 *     For the Gutenberg-Richter tables amplitudes are measured in micrometers
 *         Q(d,h) = QGR(d,h) - 3
 *     For the Veith-Clawson and Murphy and Barker tables amplitudes are
 *     measured in nanometers
 *         Q(d,h) = QVC(d,h) - 3
 *  Input Arguments:
 *     delta - epicentral distance
 *     depth - source depth
 *     isQ   - use magnitude attenuation table? (0/1/2)
 *             0 - do not use
 *             1 - Gutenberg-Richter
 *             2 - Veith-Clawson
 *     magQ  - pointer to MAGQ Q(d,h) table
 *  Return:
 *     q     - Q(d,h)
 *  Called by:
 *     calc_sta_mb
 *  Calls:
 *     bilinear_int
 */
static double get_magQ(double delta, double depth, int isQ, MAGQ *magQ)
{
    double q = 0.;
/*
 *  check for validity
 */
    if (!isQ ||
        depth < magQ->mindepth || depth > magQ->maxdepth ||
        delta < magQ->mindist  || delta > magQ->maxdist)
        return q;
/*
 *  bilinear interpolation on Q(d,h)
 */
    else
        q = bilinear_int(delta, depth, magQ->num_dists, magQ->num_depths,
                         magQ->deltas, magQ->depths, magQ->q);
/*
 *  Gutenberg-Richter Q(d,h) is valid for amplitudes in micrometer
 */
    if (isQ == 1) q -= 3;
    return q;
}

/*
 *  Title:
 *     calc_sta_ms
 *  Synopsis:
 *     Calculates MS for a single reading.
 *     Vanek, J., A. Zatopek, V. Karnik, N.V. Kondorskaya, Y.V. Riznichenko,
 *         Y.F. Savarensky, S.L. Solovev and N.V. Shebalin, 1962,
 *         Standardization of magnitude scales,
 *         Bull. (Izvest.) Acad. Sci. USSR Geophys. Ser., 2, 108-111.
 *     Amplitude MS = log(A/T) + 1.66 * log(d) + 0.3 (Prague formula)
 *     First, find max(A/T) for Z component and calculate vertical MS
 *         MS_Z = log(max(A_z/T_z)) + 1.66 * log(d) + 0.3
 *     Second, find max(A/T) for E and N components within +/- pertol seconds
 *         of Z period and calculate horizontal MS
 *         max(A/T)_h = sqrt(max(A_e/T_e)^2 + max(A_n/T_n)^2)
 *         max(A/T)_h = sqrt(2 * max(A_e/T_e)^2)      if N does not exist
 *         max(A/T)_h = sqrt(2 * max(A_n/T_n)^2)      if E does not exist
 *         MS_H = log(max(A/T)_h) + 1.66 * log(d) + 0.3
 *     Reading MS
 *         MS = (MS_H + MS_Z) / 2  if MS_Z and MS_H exist
 *         MS = MS_H               if MS_Z does not exist
 *         MS = MS_Z               if MS_H does not exist
 *  Input Arguments:
 *     start - starting index in a reading
 *     n     - number of phases in the reading
 *     p[]   - array of phase structures
 *     mszh  - pointer to MS vertical/horizontal magnitude structure
 *  Return:
 *     reading_ms - MS for this reading
 *  Called by:
 *     netmag
 */
static double calc_sta_ms(int start, int n, PHAREC p[], int mtypeid,
                          MSZH *mszh)
{
    extern double surf_mag_min_dist;                    /* From config file */
    extern double surf_mag_max_dist;                    /* From config file */
    extern double surf_mag_min_per;                     /* From config file */
    extern double surf_mag_max_per;                     /* From config file */
    extern char ms_phase[MAXNUMPHA][PHALEN];             /* From model file */
    extern int ms_phase_num;                             /* From model file */
    extern double pertol;         /* MSH period tolerance around MSZ period */

    double reading_ms = -999.;
    double apert = 0., apertn = 0., maxatz = 0., maxate = 0., maxatn = 0.;
    double ms_h = 0., ms_z = 0., logat = 0., delta = 0.;
    double minper = surf_mag_min_per, maxper = surf_mag_max_per;
    int ind_phaz = 0, ind_phae = 0, ind_phan = 0;
    int ind_ampz = 0, ind_ampe = 0, ind_ampn = 0;
    int i, j;
/*
 *  Calculate reading MS from max(A/T)
 */
    ind_phaz = ind_phae = ind_phan = -1;
    ind_ampz = ind_ampe = ind_ampn = -1;
    maxatz = maxate = maxatn = -999.;
    ms_z = ms_h = NULLVAL;
    delta = p[start].delta;
/*
 *  First, find max(A/T) for Z component
 */
    for (i = start; i < n; i++) {
/*
 *      Only need look at phases with amplitudes
 */
        if (p[i].numamps == 0)
            continue;
        if (p[i].purged)       /* used by iscloc_search */
            continue;
/*
 *      Only consider phases contributing to MS
 */
        for (j = 0; j < ms_phase_num; j++)
            if (streq(p[i].phase, ms_phase[j]))
                break;
        if (j == ms_phase_num)
            continue;
/*
 *      ignore observations outside the MS distance range
 */
        if (delta < surf_mag_min_dist || delta > surf_mag_max_dist)
            continue;
/*
 *      Loop over amplitudes
 */
        for (j = 0; j < p[i].numamps; j++) {
/*
 *          ignore amplitudes outside the period range
 */
            if (p[i].a[j].per < surf_mag_min_per ||
                p[i].a[j].per > surf_mag_max_per)
                continue;
/*
 *          ignore null amplitudes
 */
            if (fabs(p[i].a[j].amp) < DEPSILON || p[i].a[j].amp == NULLVAL)
                continue;
/*
 *          ignore null periods
 */
            if (fabs(p[i].a[j].per) < DEPSILON || p[i].a[j].per == NULLVAL)
                continue;
/*
 *          ignore non-vertical components
 */
            if (p[i].a[j].comp != 'Z')
                continue;
/*
 *          keep track of maximum(A/T)
 */
            apert = p[i].a[j].amp / p[i].a[j].per;
            if (apert > maxatz) {
                maxatz = apert;
                ind_ampz = j;
                ind_phaz = i;
            }
/*
 *          amplitude magnitude using Prague formula
 */
            if (p[i].phase[0] == 'I')
                p[i].a[j].amp *= 1000.;
            logat = log10(p[i].a[j].amp / p[i].a[j].per);
            p[i].a[j].magnitude = logat + 1.66 * log10(delta) + 0.3;
            p[i].a[j].ampdef = 0;
            p[i].a[j].mtypeid = mtypeid;
            strcpy(p[i].a[j].magtype, "MS");
            if (verbose > 2) {
                fprintf(logfp, "          i=%2d j=%2d comp=%c amp=%.2f ",
                        i, j, p[i].a[j].comp, p[i].a[j].amp);
                fprintf(logfp, "per=%.2f logat%.2f magnitude: %.2f\n",
                        p[i].a[j].per, logat, p[i].a[j].magnitude);
            }
        }
    }
/*
 *  MS_Z
 */
    if (ind_phaz > -1) {
        minper = p[ind_phaz].a[ind_ampz].per - pertol;
        maxper = p[ind_phaz].a[ind_ampz].per + pertol;
        p[ind_phaz].a[ind_ampz].ampdef = 1;
        ms_z = p[ind_phaz].a[ind_ampz].magnitude;
        if (verbose > 1)
            fprintf(logfp, "        rdid=%d %-6s delta=%.2f MS_z=%.2f\n",
                    p[ind_phaz].rdid, p[ind_phaz].sta, delta, ms_z);
    }
/*
 *
 *  Second, find max(A/T) for E and N components
 *
 */
    for (i = start; i < n; i++) {
/*
 *      Only need look at phases with amplitudes
 */
        if (p[i].numamps == 0)
            continue;
        if (p[i].purged)       /* used by iscloc_search */
            continue;
/*
 *      Only consider phases contributing to MS
 */
        for (j = 0; j < ms_phase_num; j++)
            if (streq(p[i].phase, ms_phase[j]))
                break;
        if (j == ms_phase_num)
            continue;
/*
 *      ignore observations outside the MS distance range
 */
        if (delta < surf_mag_min_dist || delta > surf_mag_max_dist)
            continue;
/*
 *      Loop over amplitudes
 */
        for (j = 0; j < p[i].numamps; j++) {
/*
 *          ignore amplitudes outside the period range
 */
            if (p[i].a[j].per < minper || p[i].a[j].per > maxper)
                continue;
/*
 *          ignore null amplitudes
 */
            if (fabs(p[i].a[j].amp) < DEPSILON || p[i].a[j].amp == NULLVAL)
                continue;
/*
 *          ignore non-horizontal components
 */
            if (p[i].a[j].comp != 'N' && p[i].a[j].comp != 'E')
                continue;
/*
 *          keep track of maximum(A/T)
 */
            if (p[i].a[j].comp == 'E') {
                apert = p[i].a[j].amp / p[i].a[j].per;
                if (apert > maxate) {
                    maxate = apert;
                    ind_ampe = j;
                    ind_phae = i;
                }
            }
            if (p[i].a[j].comp == 'N') {
                apertn = p[i].a[j].amp / p[i].a[j].per;
                if (apertn > maxatn) {
                    maxatn = apertn;
                    ind_ampn = j;
                    ind_phan = i;
                }
            }
/*
 *          amplitude magnitude using Prague formula
 */
            if (p[i].phase[0] == 'I')
                p[i].a[j].amp *= 1000.;
            logat = log10(p[i].a[j].amp / p[i].a[j].per);
            p[i].a[j].magnitude = logat + 1.66 * log10(delta) + 0.3;
            p[i].a[j].ampdef = 0;
            p[i].a[j].mtypeid = mtypeid;
            strcpy(p[i].a[j].magtype, "MS");
            if (verbose > 2) {
                fprintf(logfp, "          i=%2d j=%2d comp=%c amp=%.2f ",
                        i, j, p[i].a[j].comp, p[i].a[j].amp);
                fprintf(logfp, "per=%.2f logat%.2f magnitude: %.2f\n",
                        p[i].a[j].per, logat, p[i].a[j].magnitude);
            }
        }
    }
/*
 *  MS_H
 */
    if (ind_phan > -1 && ind_phae > -1) {
/*
 *      ms_h from E and N components
 */
        p[ind_phan].a[ind_ampn].ampdef = 1;
        p[ind_phae].a[ind_ampe].ampdef = 1;
        apert = sqrt(maxatn * maxatn + maxate * maxate);
        ms_h = log10(apert) + 1.66 * log10(delta) + 0.3;
        if (verbose)
            fprintf(logfp, "        rdid=%d %-6s delta=%.2f MS_h=%.2f\n",
                    p[ind_phan].rdid, p[ind_phan].sta, delta, ms_h);
    }
    else if (ind_phae > -1) {
/*
 *      ms_h from E component
 */
        p[ind_phae].a[ind_ampe].ampdef = 1;
        apert = sqrt(2. * maxate * maxate);
        ms_h = log10(apert) + 1.66 * log10(delta) + 0.3;
        if (verbose)
            fprintf(logfp, "        rdid=%d %-6s delta=%.2f MS_h=%.2f\n",
                    p[ind_phae].rdid, p[ind_phae].sta, delta, ms_h);
    }
    else if (ind_phan > -1) {
/*
 *      ms_h from N component
 */
        p[ind_phan].a[ind_ampn].ampdef = 1;
        apert = sqrt(2. * maxatn * maxatn);
        ms_h = log10(apert) + 1.66 * log10(delta) + 0.3;
        if (verbose)
            fprintf(logfp, "        rdid=%d %-6s delta=%.2f MS_h=%.2f\n",
                    p[ind_phan].rdid, p[ind_phan].sta, delta, ms_h);
    }
/*
 *
 *  reading MS depending on the MS horizontal and vertical components
 *
 */
    if (ms_h != NULLVAL) {
        mszh->rdid = p[start].rdid;
        mszh->msh = ms_h;
        mszh->mshdef = 1;
        if (ms_z != NULLVAL) {
            reading_ms = (ms_h + ms_z) / 2.;
            mszh->msz = ms_z;
            mszh->mszdef = 1;
        }
        else {
            reading_ms = ms_h;
            mszh->msz = ms_z;
            mszh->mszdef = 0;
        }
    }
    else if (ms_z != NULLVAL) {
        reading_ms = ms_z;
        mszh->rdid = p[start].rdid;
        mszh->msz = ms_z;
        mszh->mszdef = 1;
        mszh->msh = ms_h;
        mszh->mshdef = 0;
    }
    return reading_ms;
}

/*
 *  Title:
 *     read_magQ
 *  Synopsis:
 *     Reads magnitude attenuation Q(d,h) table from file.
 *  Input Arguments:
 *     fname - pathname for magnitude attenuation table
 *     magq  - pointer to MAGQ structure
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     read_data_files
 *  Calls:
 *     skipcomments, alloc_matrix, Free
 */
int read_magQ(char *filename, MAGQ *magq)
{
    FILE *fp;
    char buf[LINLEN];
    int ndis = 0, ndep = 0;
    int i, j;
    double x = 0.;
/*
 *  open magnitude attenuation file
 */
    if ((fp = fopen(filename, "r")) == NULL) {
        fprintf(logfp, "read_magQ: cannot open %s\n", filename);
        fprintf(errfp, "read_magQ: cannot open %s\n", filename);
        errorcode = 2;
        return 1;
    }
    while (fgets(buf, LINLEN, fp)) {
/*
 *      distance samples
 */
        if (strstr(buf, "DISTANCE")) {
            skipcomments(buf, fp);
            sscanf(buf, "%d", &ndis);
            if ((magq->deltas = (double *)calloc(ndis, sizeof(double))) == NULL) {
                fprintf(logfp, "read_magQ: cannot allocate memory\n");
                fprintf(errfp, "read_magQ: cannot allocate memory\n");
                errorcode = 1;
                return 1;
            }
            magq->num_dists = ndis;
            for (i = 0; i < ndis; i++) {
                fscanf(fp, "%lf", &x);
                magq->deltas[i] = x;
            }
            magq->mindist = magq->deltas[0];
            magq->maxdist = magq->deltas[ndis - 1];
        }
/*
 *      depth samples
 */
        if (strstr(buf, "DEPTH")) {
            skipcomments(buf, fp);
            sscanf(buf, "%d", &ndep);
            if ((magq->depths = (double *)calloc(ndep, sizeof(double))) == NULL) {
                fprintf(logfp, "read_magQ: cannot allocate memory\n");
                fprintf(errfp, "read_magQ: cannot allocate memory\n");
                Free(magq->deltas);
                errorcode = 1;
                return 1;
            }
            magq->num_depths = ndep;
            for (i = 0; i < ndep; i++) {
                fscanf(fp, "%lf", &x);
                magq->depths[i] = x;
            }
            magq->mindepth = magq->depths[0];
            magq->maxdepth = magq->depths[ndep - 1];
        }
/*
 *      Q(d,h) table
 */
        if (strstr(buf, "MAGNITUDE")) {
            skipcomments(buf, fp);
            sscanf(buf, "%d %d", &i, &j);
            if ((magq->q = alloc_matrix(ndis, ndep)) == NULL) {
                fprintf(logfp, "read_magQ: cannot allocate memory\n");
                fprintf(errfp, "read_magQ: cannot allocate memory\n");
                Free(magq->deltas);
                Free(magq->depths);
                errorcode = 1;
                return 1;
            }
            for (i = 0; i < ndis; i++) {
                for (j = 0; j < ndep; j++) {
                    fscanf(fp, "%lf", &x);
                    magq->q[i][j] = x;
                }
            }
        }
    }
    fclose(fp);
    return 0;
}

/*  EOF  */
