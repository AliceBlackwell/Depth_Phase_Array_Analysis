#include "iscloc.h"
extern int verbose;
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;
extern int numagencies;
extern char agencies[MAXBUF][AGLEN];
extern int numsta;
char isf_error[LINLEN];                      /* buffer for ISF parse errors */

/*
 * Functions:
 *    read_stafile
 *    read_stafile_isf2
 *    read_isf
 *    write_isf
 *    check_int
 */

/*
 * Local functions
 */
static int parse_event_id(char *line, int *evid, char *eventid);
static int parse_origin(char *line, int *yyyy, int *mm, int *dd,
                        int *hh, int *mi, int *ss, int *msec, char *timfix,
                        double *stime, double *sdobs, double *lat, double *lon,
                        char *epifix, double *smaj, double *smin, int *strike,
                        double *depth, char *depfix, double *sdepth, int *ndef,
                        int *nsta, int *gap, double *mindist, double *maxdist,
                        char *etype, char *author, char *origid, int *hypid,
                        char *hrep, double *depdp, double *dpderr, int *sgap);
static int parse_phase(char *line, char *sta, double *dist, char *phase,
                       int *hh, int *mi, int *ss, int *msec, char *arrid,
                       double *azim, double *slow, double *amp, double *per,
                       char *sp_fm, char *detchar, char *agy, char *deploy,
                       char *lcn, char *auth, char *rep, char *pch, char *ach,
                       char *lp_fm, double *snr);
static int sta_compare(const void *sta1, const void *sta2);
static int fdsn_compare(const void *sta1, const void *sta2);
static int find_fdsn_index(int nsta, STAREC stalist[], char *fdsn);
static int find_sta_index(int nsta, STAREC stalist[], char *sta);

static void write_origin(FILE *fp, char *hday, char *htim, char timfix,
                         double stime, double sdobs, double lat, double lon,
                         char epifix, double smaj, double smin, int strike,
                         double depth, char depfix, double sdepth, int ndef,
                         int nsta, int gap, double mindist, double maxdist,
                         char antype, char loctype, char *etype,
                         char *author, char *origid, char *rep,
                         double depdp, double dpderr, int sgap);
static void write_phase(FILE *fp, char *sta, double dist, double esaz,
                        char *phase, char *htim, double timeres,
                        double azim, double slow, char timedef, double snr,
                        double amp, double per, char sp_fm, char detchar,
                        char *magtype, double mag, char *arrid,
                        char *agency, char *deploy, char *lcn, char *auth,
                        char *rep, char *pch, char *ach, char lp_fm);

static int partline(char *part, char *line, int offset, int numchars);
static int check_whole(char *substr);
static int all_blank(char *s);
static int check_float(char *substr);

/*
 *  Title:
 *     read_stafile
 *  Synopsis:
 *     Reads stations from ISF stafile into an array.
 *  Input Arguments:
 *     stationlist[] - array of structures for station information.
 *  Returns:
 *     0/1 on success/error
 *  Called by:
 *     main
 *  Calls:
 *     sta_compare
 */
int read_stafile(STAREC *stationlistp[])
{
    extern char isf_stafile[];        /* From config file */
    FILE *fp;
    char line[LINLEN], *s;
    STAREC *slist = (STAREC *)NULL;
    int i, k;
    double elev = 0.;
/*
 *  Open station file an get number of stations
 */
    if ((fp = fopen(isf_stafile, "r")) == NULL) {
        fprintf(errfp, "read_stafile: Cannot open %s\n", isf_stafile);
        return 1;
    }
    i = 0;
    while (fgets(line, LINLEN, fp)) i++;
    numsta = i;
    rewind(fp);
/*
 *  memory allocation
 */
    if ((*stationlistp = (STAREC *)calloc(numsta, sizeof(STAREC))) == NULL) {
        fprintf(errfp, "read_stafile: cannot allocate memory\n");
        fclose(fp);
        errorcode = 1;
        return 1;
    }
    slist = *stationlistp;
/*
 *  read station info
 */
    i = 0;
    while (fgets(line, LINLEN, fp)) {
        k = 0;
        s = strtok(line, ", ");
        strcpy(slist[i].sta, s);
        strcpy(slist[i].agency, "FDSN");
        strcpy(slist[i].deploy, "IR");
        strcpy(slist[i].lcn, "--");
        while ((s = strtok(NULL, ", ")) != NULL) {
            switch (k) {
                case 0:
                    strcpy(slist[i].altsta, s);
                    sprintf(slist[i].fdsn, "%s.%s.%s.%s",
                            slist[i].agency, slist[i].deploy,
                            slist[i].sta, slist[i].lcn);
                    break;
                case 1:
                    slist[i].lat = atof(s);
                    break;
                case 2:
                    slist[i].lon = atof(s);
                    break;
                case 3:
                    slist[i].elev = NULLVAL;
                    elev = atof(s);
                    if (elev > -9999) slist[i].elev = elev;
                    break;
                default:
                    break;
            }
            k++;
        }
        i++;
    }
    fclose(fp);
    qsort(slist, numsta, sizeof(STAREC), sta_compare);
    return 0;
}

/*
 *  Title:
 *     read_stafile_isf2
 *  Synopsis:
 *     Reads stations from ISF2 stafile into an array.
 *  Input Arguments:
 *     stationlist[] - array of structures for station information.
 *  Returns:
 *     0/1 on success/error
 *  Called by:
 *     main
 *  Calls:
 *     fdsn_compare
 */
int read_stafile_isf2(STAREC *stationlistp[])
{
    extern char isf_stafile[];        /* From config file */
    FILE *fp;
    char line[LINLEN], *s, *p;
    STAREC *slist = (STAREC *)NULL;
    int i, j, k;
    double elev = 0.;
/*
 *  Open station file an get number of stations
 */
    if ((fp = fopen(isf_stafile, "r")) == NULL) {
        fprintf(errfp, "read_stafile_isf2: Cannot open %s\n", isf_stafile);
        return 1;
    }
    i = 0;
    while (fgets(line, LINLEN, fp)) {
        if (strstr(line, "<EOE>"))
            i++;
    }
    numsta = i;
    rewind(fp);
/*
 *  memory allocation
 */
    if ((*stationlistp = (STAREC *)calloc(numsta, sizeof(STAREC))) == NULL) {
        fprintf(errfp, "read_stafile_isf2: cannot allocate memory\n");
        fclose(fp);
        errorcode = 1;
        return 1;
    }
    slist = *stationlistp;
/*
 *  read station info
 */
    i = 0;
    k = 1;
    while (fgets(line, LINLEN, fp)) {
        if (strstr(line, "<EO"))
            continue;
        s = strtok(line, ":");
        strcpy(slist[i].fdsn, s);
        s = strtok(NULL, ":");
        p = strtok(s, " ");
        slist[i].lat = atof(p);
        p = strtok(NULL, " ");
        slist[i].lon = atof(p);
        p = strtok(NULL, " ");
        slist[i].elev = NULLVAL;
        elev = atof(p);
        if (elev > -9999 && elev < 9999) slist[i].elev = elev;
        strcpy(s, slist[i].fdsn);
        p = strtok(s, ".");
        strcpy(slist[i].agency, p);
        p = strtok(NULL, ".");
        strcpy(slist[i].deploy, p);
        p = strtok(NULL, ".");
        strcpy(slist[i].sta, p);
        p = strtok(NULL, ".");
        strcpy(slist[i].lcn, p);
        sprintf(slist[i].altsta, "%05d", k);

        k++;
/*
 *      co-located stations get the same altsta for nearest-neighbour sort
 */
        for (j = 0; j < i; j++) {
            if (fabs(slist[i].lat - slist[j].lat) < DEPSILON &&
                fabs(slist[i].lon - slist[j].lon) < DEPSILON) {
                strcpy(slist[i].altsta, slist[j].altsta);
                k--;
                break;
            }
        }
        if (verbose > 5)
            fprintf(logfp, "%s %6.3f %7.3f %.1f %s\n",
                    slist[i].fdsn, slist[i].lat, slist[i].lon, slist[i].elev,
                    slist[i].altsta);
        i++;
    }
    fclose(fp);
    qsort(slist, numsta, sizeof(STAREC), fdsn_compare);
    fprintf(logfp, "%d stations, %d distinct station locations\n",
            numsta, k);
    return 0;
}

/*
 *
 * sta_compare: compares two stalist records based on sta code
 *
 */
static int sta_compare(const void *sta1, const void *sta2)
{
    return strcmp(((STAREC *)sta1)->sta, ((STAREC *)sta2)->sta);
}

/*
 *
 * fdsn_compare: compares two stalist records based on fdsn code
 *
 */
static int fdsn_compare(const void *sta1, const void *sta2)
{
    return strcmp(((STAREC *)sta1)->fdsn, ((STAREC *)sta2)->fdsn);
}

/*
 *  Title:
 *     find_sta_index
 *  Synopsis:
 *     Returns index of a station in the stalist array
 *  Input Arguments:
 *     nsta      - number of distinct stations
 *     stalist[] - array of starec structures
 *     sta       - station to find
 *  Return:
 *     station index or -1 on error
 *  Called by:
 *     read_isf
 */
static int find_sta_index(int nsta, STAREC stalist[], char *sta)
{
    int klo = 0, khi = nsta - 1, k = 0, i;
    if (nsta > 2) {
        while (khi - klo > 1) {
            k = (khi + klo) >> 1;
            if ((i = strcmp(stalist[k].sta, sta)) == 0)
                return k;
            else if (i > 0)
                khi = k;
            else
                klo = k;
        }
        if (khi == 1) {
            k = 0;
            if (streq(sta, stalist[k].sta)) return k;
            else return -1;
        }
        if (klo == nsta - 2) {
            k = nsta - 1;
            if (streq(sta, stalist[k].sta)) return k;
            else return -1;
        }
    }
    else if (nsta == 2) {
        if (streq(sta, stalist[0].sta)) return 0;
        else if (streq(sta, stalist[1].sta)) return 1;
        else return -1;
    }
    else {
        if (streq(sta, stalist[0].sta)) return 0;
        else return -1;
    }
    return -1;
}

/*
 *  Title:
 *     find_fdsn_index
 *  Synopsis:
 *     Returns index of an IR2 station in the stalist array
 *  Input Arguments:
 *     nsta      - number of distinct stations
 *     stalist[] - array of starec structures
 *     fdsn      - station to find
 *  Return:
 *     station index or -1 on error
 *  Called by:
 *     read_isf
 */
static int find_fdsn_index(int nsta, STAREC stalist[], char *fdsn)
{
    int klo = 0, khi = nsta - 1, k = 0, i;
    if (nsta > 2) {
        while (khi - klo > 1) {
            k = (khi + klo) >> 1;
            if ((i = strcmp(stalist[k].fdsn, fdsn)) == 0)
                return k;
            else if (i > 0)
                khi = k;
            else
                klo = k;
        }
        if (khi == 1) {
            k = 0;
            if (streq(fdsn, stalist[k].fdsn)) return k;
            else return -1;
        }
        if (klo == nsta - 2) {
            k = nsta - 1;
            if (streq(fdsn, stalist[k].fdsn)) return k;
            else return -1;
        }
    }
    else if (nsta == 2) {
        if (streq(fdsn, stalist[0].fdsn)) return 0;
        else if (streq(fdsn, stalist[1].fdsn)) return 1;
        else return -1;
    }
    else {
        if (streq(fdsn, stalist[0].fdsn)) return 0;
        else return -1;
    }
    return -1;
}

/*
 *  Title:
 *    read_isf
 *  Synopsis:
 *    Reads next event from an ISF bulletin format file.
 *    Assumes that the prime hypocentre is the last one in the origin block.
 *    Allocates memory to hypocentre, phase and magnitude structures.
 *  Input Arguments:
 *    infile    - file pointer to read from
 *    isf       - ISF version {1,2}
 *    ep        - pointer to event info
 *    pp        - pointer to phase structures
 *    hp        - pointer to hypocentre structures
 *    stamag_mb - pointer to mb stamag structures
 *    stamag_ms - pointer to MS stamag structures
 *    rdmag_mb  - pointer to mb reading mag structures
 *    rdmag_ms  - pointer to MS reading mag structures
 *    mszh      - pointer to MS horizontal and vertical components
 *    magbloc   - reported magnitudes from ISF input file
 *  Return:
 *    0/1 on success/error.
 *  Called by:
 *     main
 *  Calls:
 *    parse_event_id, all_blank, parse_origin, parse_phase, epoch_time_isf,
 *    find_sta_index, find_fdsn_index, sort_phaserec_isf, print_pha
 */
int read_isf(FILE *infile, int isf, EVREC *ep, HYPREC *hp[], PHAREC *pp[],
             STAREC stationlist[], STAMAG *stamag_mb[], STAMAG *stamag_ms[],
             RDMAG *rdmag_mb[], RDMAG *rdmag_ms[], MSZH *mszh[], char *magbloc)
{
    HYPREC *h = (HYPREC *)NULL;
    PHAREC *p = (PHAREC *)NULL;
    char line[LINLEN], mline[LINLEN], eventid[AGLEN];
    int yyyy = 0, mm = 0, dd = 0, hh = 0, mi = 0, ss = 0, msec = 0;
    int evid = 0, rdid = 0;
    char prev_sta[STALEN];
    int found = 0;
    int i, j, m, nstas = 0;
    int first_P = 0;
/*
 *  Arrays for storing hypocentre values as they are read in.
 */
    char timfix[MAXHYP], epifix[MAXHYP], depfix[MAXHYP];
    double hyp_time[MAXHYP], lat[MAXHYP], lon[MAXHYP], depth[MAXHYP];
    double stime[MAXHYP], sdepth[MAXHYP], sdobs[MAXHYP];
    double smajax[MAXHYP], sminax[MAXHYP];
    int strike[MAXHYP], ndef[MAXHYP], nsta[MAXHYP], gap[MAXHYP], sgap[MAXHYP];
    double mindist[MAXHYP], maxdist[MAXHYP], depdp[MAXHYP], dpderr[MAXHYP];
    char etype[MAXHYP][3], author[MAXHYP][AGLEN], hrep[MAXHYP][AGLEN];
    char origid[MAXHYP][15];
    int hypid[MAXHYP];
/*
 *   Arrays for reading in phase data.
 */
    char sta[MAXPHA][STALEN], agy[MAXPHA][AGLEN], deploy[MAXPHA][AGLEN];
    char auth[MAXPHA][AGLEN], rep[MAXPHA][AGLEN], lcn[MAXPHA][3];
    char phase[MAXPHA][PHALEN], ach[MAXPHA][5], pch[MAXPHA][5];
    double delta[MAXPHA], pha_time[MAXPHA], azim[MAXPHA], slow[MAXPHA];
    double amp[MAXPHA], per[MAXPHA], snr[MAXPHA];
    char sp_fm[MAXPHA], lp_fm[MAXPHA], detchar[MAXPHA], arrid[MAXPHA][15];
/*
 *  Parse ISF file
 *
 *
 *  find next event
 */
    ep->eventid[0] = '\0';
    ep->evid = NULLVAL;
    while (fgets(line, LINLEN, infile)) {
        if (parse_event_id(line, &evid, eventid) == 0) {
            ep->evid = evid;
            strcpy(ep->eventid, eventid);
            break;
        }
    }
    if (ep->evid == NULLVAL) {
        if (!feof(infile))
            fprintf(errfp, "read_isf: no event line found\n");
        return 1;
    }
    if (verbose) fprintf(logfp, "    found event id %d\n", ep->evid);
/*
 *  find origin block
 */
    found = 0;
    while (fgets(line, LINLEN, infile)) {
        if (strstr(line, "Latitude Longitude")) {
            found = 1;
            break;
        }
    }
    if (!found) {
        fprintf(errfp, "read_isf: no origin block found\n");
        return 1;
    }
/*
 *  parse origin block
 */
    i = 0;
    while (fgets(line, LINLEN, infile)) {
/*
 *      a blank line terminates the origin block
 */
        if (all_blank(line))
            break;
/*
 *      skip ISF comment lines
 */
        if (!strncmp(line, " (", 2))
            continue;
/*
 *      read origin line
 */
        if (parse_origin(line, &yyyy, &mm, &dd, &hh, &mi, &ss, &msec,
                         &timfix[i], &stime[i], &sdobs[i], &lat[i], &lon[i],
                         &epifix[i], &smajax[i], &sminax[i], &strike[i],
                         &depth[i], &depfix[i], &sdepth[i], &ndef[i], &nsta[i],
                         &gap[i], &mindist[i], &maxdist[i], etype[i],
                         author[i], origid[i], &hypid[i], hrep[i],
                         &depdp[i], &dpderr[i], &sgap[i]) == 0) {
            if (msec == NULLVAL) msec = 0;
            hyp_time[i++] = epoch_time_isf(yyyy, mm, dd, hh, mi, ss, msec);
        }
        else
            fprintf(errfp, "read_isf: origin: %s\n", isf_error);
    }
    if ((ep->numhyps = i) == 0) {
        fprintf(errfp, "read_isf: no hypocentres found\n");
        return 1;
    }
    if (verbose) fprintf(logfp, "    %d hypocentres read\n", ep->numhyps);
/*
 *  find magnitude (if any) and phase blocks
 */
    found = 0;
    while (fgets(line, LINLEN, infile)) {
        if (strstr(line, "Magnitude  Err Nsta")) {
/*
 *          accumulate magnitude block in a string for output
 */
            strcpy(magbloc, "");
            while(fgets(mline, LINLEN, infile)) {
/*
 *              a blank line terminates the magnitude block
 */
                if (all_blank(mline))
                    break;
                strcat(magbloc, mline);
            }
        }
        if (strstr(line, "EvAz Phase")) {
            found = 1;
            break;
        }
    }
    if (!found) {
        fprintf(errfp, "read_isf: no phase block found\n");
        return 1;
    }
/*
 *  parse phase block
 */
    i = 0;
    while (fgets(line, LINLEN, infile)) {
/*
 *      a blank line terminates the phase block
 */
        if (all_blank(line))
            break;
/*
 *      skip ISF comment lines
 */
        if (!strncmp(line, " (", 2))
            continue;
/*
 *      read phase line
 */
        if (parse_phase(line, sta[i], &delta[i], phase[i],
                        &hh, &mi, &ss, &msec, arrid[i], &azim[i], &slow[i],
                        &amp[i], &per[i], &sp_fm[i], &detchar[i],
                        agy[i], deploy[i], lcn[i], auth[i], rep[i],
                        pch[i], ach[i], &lp_fm[i], &snr[i]) == 0) {
/*
 *          arrival epoch time: take yyyy,mm,dd from last (prime) hypocentre
 */
            if (hh == NULLVAL)
                pha_time[i] = NULLVAL;
            else {
                if (msec == NULLVAL) msec = 0.;
                pha_time[i] = epoch_time_isf(yyyy, mm, dd, hh, mi, ss, msec);
/*
 *              Only way phase could be before origin is if day changes
 */
//                if (pha_time[i] < hyp_time[ep->numhyps-1])
//                    pha_time[i] += 86400;
            }
            if (delta[i] == NULLVAL) {
                fprintf(errfp, "read_isf: phases must have delta\n");
                return 1;
            }
            i++;
        }
        else
            fprintf(errfp, "read_isf: phase: %s\n", isf_error);
    }
    if ((ep->numphas = i) == 0) {
        fprintf(errfp, "read_isf: no phases read\n");
        return 1;
    }
    if (verbose) fprintf(logfp, "    %d phases read\n", ep->numphas);
/*
 *  skip rest of the lines belonging to the event (NEIC PDE ISF2)
 */
    if (isf == 2) {
        while (fgets(line, LINLEN, infile)) {
            if (strstr(line, "DATA_TYPE BULLETIN"))
                break;
        }
    }
/*
 *  Memory allocations
 */
    *hp = (HYPREC *)calloc(ep->numhyps, sizeof(HYPREC));
    if (*hp == NULL) {
        fprintf(errfp, "read_isf: cannot allocate memory\n");
        errorcode = 1;
        return 1;
    }
    *pp = (PHAREC *)calloc(ep->numphas, sizeof(PHAREC));
    if (*pp == NULL) {
        fprintf(errfp, "read_isf: cannot allocate memory\n");
        Free(*hp);
        errorcode = 1;
        return 1;
    }
/*
 *  Fill in the hypocentre structures
 *  h[0] is set to the prime (last hypocentre by ISC convention)
 */
    h = *hp;
    for (i = 0, j = ep->numhyps - 1; i < ep->numhyps; i++, j--) {
        if (hypid[j] == NULLVAL) h[i].hypid = i + 1;
        else                     h[i].hypid = hypid[j];
        strcpy(h[i].origid, origid[j]);
        h[i].time  = hyp_time[j];
        h[i].lat   = lat[j];
        h[i].lon   = lon[j];
        h[i].depth = depth[j];
        h[i].ndef = ndef[j];
        h[i].ndefsta = nsta[j];
        h[i].mindist = mindist[j];
        h[i].maxdist = maxdist[j];
        h[i].azimgap = gap[j];
        h[i].sgap = sgap[j];
        strcpy(h[i].etype, etype[j]);
        strcpy(h[i].agency, author[j]);
        strcpy(h[i].rep, hrep[j]);
        h[i].sdobs = sdobs[j];
        h[i].stime = stime[j];
        h[i].sdepth = sdepth[j];
        h[i].sminax = sminax[j];
        h[i].smajax = smajax[j];
        h[i].strike = strike[j];
        h[i].depdp = depdp[j];
        h[i].dpderr = dpderr[j];
        h[i].depfix = h[i].epifix = h[i].timfix = 0;
    }
    numagencies = 1;
    strcpy(agencies[0], h[0].agency);
/*
 *  Fill in the phase structures
 */
    p = *pp;
    for (j = 0, i = 0; i < ep->numphas; i++) {
        p[j].hypid = h[0].hypid;
        p[j].phid  = i + 1;
        strcpy(p[j].arrid, arrid[i]);
        strcpy(p[j].rep_phase, phase[i]);
        strcpy(p[j].auth, auth[i]);
        strcpy(p[j].rep, rep[i]);
        strcpy(p[j].agency, agy[i]);
        strcpy(p[j].deploy, deploy[i]);
        strcpy(p[j].lcn, lcn[i]);
        strcpy(p[j].sta, sta[i]);
        sprintf(p[j].fdsn, "%s.%s.%s.%s",
                p[j].agency, p[j].deploy, p[j].sta, p[j].lcn);
        if (isf == 2) {
            m = find_fdsn_index(numsta, stationlist, p[j].fdsn);
            if (m < 0) {
                fprintf(errfp, "read_isf: missing station %s\n", p[j].fdsn);
                fprintf(logfp, "read_isf: missing station %s\n", p[j].fdsn);
                continue;
            }
        }
        else {
            m = find_sta_index(numsta, stationlist, p[j].sta);
            if (m < 0) {
                fprintf(errfp, "read_isf: missing station %s\n", p[j].sta);
                fprintf(logfp, "read_isf: missing station %s\n", p[j].sta);
                continue;
            }
        }
        strcpy(p[j].prista, stationlist[m].altsta);
        p[j].delta = delta[i];
        p[j].time = pha_time[i];
        strcpy(p[j].pch, pch[i]);
        p[j].snr = snr[i];
        p[j].slow = slow[i];
        p[j].azim = azim[i];
        if (amp[j] == NULLVAL)
            p[j].numamps = 0;
        else {
            p[j].numamps = 1;
            p[j].a[0].amp = amp[i];
            p[j].a[0].per = per[i];
            strcpy(p[j].a[0].ach, ach[i]);
            if (ach[i][2] == 'Z' || ach[i][2] == 'E' || ach[i][2] == 'N')
                p[j].a[0].comp = ach[i][2];
            else {
                p[j].a[0].comp = ' ';
                if (streq(p[j].rep_phase, "LRZ") ||
                    streq(p[j].rep_phase, "LZ") ||
                    streq(p[j].rep_phase, "PMZ"))
                    p[j].a[0].comp = 'Z';
                if (streq(p[j].rep_phase, "LRE") || streq(p[j].rep_phase, "LE"))
                    p[j].a[0].comp = 'E';
                if (streq(p[j].rep_phase, "LRN") || streq(p[j].rep_phase, "LN"))
                    p[j].a[0].comp = 'N';
            }
        }
        p[j].sp_fm = sp_fm[i];
        p[j].lp_fm = lp_fm[i];
        p[j].detchar = detchar[i];
        j++;
    }
/*
 *  Sort phase structures by increasing delta, sta, time
 */
    ep->numphas = j;
    sort_phaserec_isf(ep->numphas, p);
/*
 *  Split phases into readings and use station array to assign lat, lon.
 */
    prev_sta[0] = '\0';
    rdid = 1;
    for (nstas = 0, i = 0; i < ep->numphas; i++) {
	first_P = 0;
	if ((strcmp(p[i].rep_phase, "P")==0) || (strcmp(p[i].rep_phase, "Pn")==0) || (strcmp(p[i].rep_phase, "Pg")==0) ){
	    first_P = 1;
            //fprintf(stdout, "first_P \n");
	    //fprintf(stdout, "first_P = %i\n", first_P);
	}
		
        if (strcmp(p[i].sta, prev_sta) || first_P == 1) {
            
	    p[i].rdid = rdid++;
            p[i].init = 1;
            nstas++;

	    //if (strcmp(p[i].sta, "SNAA")==0){
		//fprintf(stdout, "p[%i].rep_phase = %s \n", i, p[i].rep_phase);
		//fprintf(stdout, "p[%i].sta = %s \n\n", i, p[i].sta );
	    //}

            if (isf == 2)
                m = find_fdsn_index(numsta, stationlist, p[i].fdsn);
            else
                m = find_sta_index(numsta, stationlist, p[i].sta);
            p[i].sta_lat  = stationlist[m].lat;
            p[i].sta_lon  = stationlist[m].lon;
            p[i].sta_elev = stationlist[m].elev;
            strcpy(prev_sta, p[i].sta);
        }
        else {
            p[i].rdid     = p[i-1].rdid;
            p[i].sta_lat  = p[i-1].sta_lat;
            p[i].sta_lon  = p[i-1].sta_lon;
            p[i].sta_elev = p[i-1].sta_elev;
        }
    }
    ep->numrd = rdid - 1;
    if ((ep->numsta = nstas) == 0) {
        fprintf(errfp, "read_isf: missing stations!\n");
        Free(*pp);
        Free(*hp);
        return 1;
    }
    if ((*stamag_mb = (STAMAG *)calloc(ep->numsta, sizeof(STAMAG))) == NULL) {
        fprintf(logfp, "read_isf: cannot allocate memory\n");
        fprintf(errfp, "read_isf: cannot allocate memory\n");
        Free(*pp);
        Free(*hp);
        errorcode = 1;
        return 1;
    }
    if ((*stamag_ms = (STAMAG *)calloc(ep->numsta, sizeof(STAMAG))) == NULL) {
        fprintf(logfp, "read_isf: cannot allocate memory\n");
        fprintf(errfp, "read_isf: cannot allocate memory\n");
        Free(*stamag_mb);
        Free(*pp);
        Free(*hp);
        errorcode = 1;
        return 1;
    }
    if ((*rdmag_mb = (RDMAG *)calloc(ep->numsta, sizeof(RDMAG))) == NULL) {
        fprintf(logfp, "read_isf: evid %d: cannot allocate memory\n", ep->evid);
        fprintf(errfp, "read_isf: evid %d: cannot allocate memory\n", ep->evid);
        Free(*stamag_ms);
        Free(*stamag_mb);
        Free(*pp);
        Free(*hp);
        errorcode = 1;
        return 1;
    }
    if ((*rdmag_ms = (RDMAG *)calloc(ep->numsta, sizeof(RDMAG))) == NULL) {
        fprintf(logfp, "read_isf: evid %d: cannot allocate memory\n", ep->evid);
        fprintf(errfp, "read_isf: evid %d: cannot allocate memory\n", ep->evid);
        Free(*rdmag_mb);
        Free(*stamag_ms);
        Free(*stamag_mb);
        Free(*pp);
        Free(*hp);
        errorcode = 1;
        return 1;
    }
    if ((*mszh = (MSZH *)calloc(ep->numsta, sizeof(MSZH))) == NULL) {
        fprintf(logfp, "read_isf: evid %d: cannot allocate memory\n", ep->evid);
        fprintf(errfp, "read_isf: evid %d: cannot allocate memory\n", ep->evid);
        Free(*rdmag_ms);
        Free(*rdmag_mb);
        Free(*stamag_ms);
        Free(*stamag_mb);
        Free(*pp);
        Free(*hp);
        errorcode = 1;
        return 1;
    }
    strcpy(ep->etype, h[0].etype);
    ep->numagency = 1;
    if (!ep->depth_fix) ep->depth_fix = h[0].depfix;
    if (!ep->epi_fix)   ep->epi_fix = h[0].epifix;
    if (!ep->time_fix)  ep->time_fix= h[0].timfix;
    if (verbose) {
        fprintf(logfp, "read_isf done.\n");
        print_pha(ep->numphas, p);
    }
    return 0;
}

/*
 *  Parses an event title line.
 *     eventid could be either an integer or a string
 *     Returns 0/1 on success/error
 */
static int parse_event_id(char *line, int *evid, char *eventid)
{
    char *s;
    int i;
    i = *evid;
/*
 *  Chars 1-5:  Event
 */
    s = strtok(line, " ");
    if (streq(s, "Event") || streq(s, "EVENT")) {
        s = strtok(NULL, " ");
        strcpy(eventid, s);
        if (check_int(s))
            *evid = i + 1;
        else
            *evid = atoi(eventid);
        return 0;
    }
    else
        return 1;
}


/*
 *  Parses an origin line.
 *     Returns 0/1 on success/error
 */
static int parse_origin(char *line, int *yyyy, int *mm, int *dd,
                        int *hh, int *mi, int *ss, int *msec, char *timfix,
                        double *stime, double *sdobs, double *lat, double *lon,
                        char *epifix, double *smaj, double *smin, int *strike,
                        double *depth, char *depfix, double *sdepth, int *ndef,
                        int *nsta, int *gap, double *mindist, double *maxdist,
                        char *etype, char *author, char *origid, int *hypid,
                        char *hrep, double *depdp, double *dpderr, int *sgap)
{
    char substr[LINLEN];
/*
 *  Chars 1-4: year.
 */
    if (!partline(substr, line, 0, 4)) {
        sprintf(isf_error, "missing year: %s", line);
        return 1;
    }
    if (check_int(substr)) {
        sprintf(isf_error, "bad year: %s", line);
        return 1;
    }
    *yyyy = atoi(substr);
/*
 *  Chars 6-7: month.
 */
    if (!partline(substr, line, 5, 2)) {
        sprintf(isf_error, "missing month: %s", line);
        return 1;
    }
    if (check_int(substr)) {
        sprintf(isf_error, "bad month: %s", line);
        return 1;
    }
    *mm = atoi(substr);
/*
 *  Chars 9-10: day.
 */
    if (!partline(substr, line, 8, 2)) {
        sprintf(isf_error, "missing day: %s", line);
        return 1;
    }
    if (check_int(substr)) {
        sprintf(isf_error, "bad day: %s", line);
        return 1;
    }
    *dd = atoi(substr);
/*
*   Chars 12-13: hour.
*/
    if (!partline(substr, line, 11, 2)) {
        sprintf(isf_error, "missing hour: %s", line);
        return 1;
    }
    if (check_int(substr)) {
        sprintf(isf_error, "bad hour: %s", line);
        return 1;
    }
    *hh = atoi(substr);
/*
 *  Chars 15-16: minute.
 */
    if (!partline(substr, line, 14, 2)) {
        sprintf(isf_error, "missing minute: %s", line);
        return 1;
    }
    if (check_int(substr)) {
        sprintf(isf_error, "bad minute: %s", line);
        return 1;
    }
    *mi = atoi(substr);
/*
 *  Chars 18,19: integral second.
 */
    if (!partline(substr, line, 17, 2)) {
        sprintf(isf_error, "missing second: %s", line);
        return 1;
    }
    if (check_int(substr)) {
        sprintf(isf_error, "bad second: %s", line);
        return 1;
    }
    *ss = atoi(substr);
/*
 *  Chars 20-22: msec or spaces.
 *  Allow decimal place with no numbers after it.
 */
    if (partline(substr, line, 20, 2)) {
/*
 *      Char 20: '.' character
 */
        if (line[19] != '.') {
            sprintf(isf_error, "bad date: %s", line);
            return 1;
        }
/*
 *      Chars 21-22: 10s of msec.
 */
        if (!isdigit(line[20])) {
            sprintf(isf_error, "bad date: %s", line);
            return 1;
        }
        *msec = (line[20] - '0') * 100;
        if (isdigit(line[21]))
            *msec += (line[21] - '0') * 10;
        else if (line[21] != ' ') {
            sprintf(isf_error, "bad date: %s", line);
            return 1;
        }
    }
    else {
/*
 *      Char 20: '.' character or space
 */
        if (line[19] != '.' && line[19] != ' ') {
            sprintf(isf_error, "bad date: %s", line);
            return 1;
        }
        *msec = NULLVAL;
    }
/*
 *  Char 23: timfix - either f or space.
 */
    if (line[22] == ' ' || line[22] == 'f')
        *timfix = line[22];
    else {
        sprintf(isf_error, "bad timfix: %s", line);
        return 1;
    }
/*
 *  Chars 25-29: origin time error
 */
    if (partline(substr, line, 24, 5)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad stime: %s", line);
            return 1;
        }
        *stime = atof(substr);
    }
    else *stime = NULLVAL;
/*
 *  Chars 31-35: rms (sdobs)
 */
    if (partline(substr, line, 30, 5)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad sdobs: %s", line);
            return 1;
        }
        *sdobs = atof(substr);
    }
    else *sdobs = NULLVAL;
/*
 *  Chars 37-44: latitude
 */
    if (!partline(substr, line, 36, 8)) {
        sprintf(isf_error, "missing latitude: %s", line);
        return 1;
    }
    if (check_float(substr)) {
        sprintf(isf_error, "bad latitude: %s", line);
        return 1;
    }
    *lat = atof(substr);
/*
 *  Chars 46-54: longitude
 */
    if (!partline(substr, line, 45, 9)) {
        sprintf(isf_error, "missing longitude: %s", line);
        return 1;
    }
    if (check_float(substr)) {
        sprintf(isf_error, "bad longitude: %s", line);
        return 1;
    }
    *lon = atof(substr);
/*
 *  Char 55: epifix - either f or space.
 */
    if (line[54] == ' ' || line[54] == 'f')
        *epifix = line[54];
    else {
        sprintf(isf_error, "bad epifix: %s", line);
        return 1;
    }
/*
 *  Chars 56-60: semi-major axis of error ellipse
 */
    if (partline(substr, line, 55, 5)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad smaj: %s", line);
            return 1;
        }
        *smaj = atof(substr);
    }
    else *smaj = NULLVAL;
/*
 *  Chars 62-66: semi-minor axis of error ellipse
 */
    if (partline(substr, line, 61, 5)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad smin: %s", line);
            return 1;
        }
        *smin = atof(substr);
    }
    else *smin = NULLVAL;
/*
 *  Chars 68-70: strike
 */
    if (partline(substr, line, 67, 3)) {
        if (check_int(substr)) {
            sprintf(isf_error, "bad strike: %s", line);
            return 1;
        }
        *strike = atoi(substr);
    }
    else *strike = NULLVAL;
/*
 *  Chars 72-76: depth
 */
    if (partline(substr, line, 71, 5)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad depth: %s", line);
            return 1;
        }
        *depth = atof(substr);
    }
    else *depth = NULLVAL;
/*
 *  Char 77: depfix - either f,d, or space.
 */
    if (line[76] == ' ' || line[76] == 'f' || line[76] == 'd')
        *depfix = line[76];
    else {
        sprintf(isf_error, "bad depfix: %s", line);
        return 1;
    }
/*
 *  Chars 79-82: depth error
 */
    if (partline(substr, line, 78, 4)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad sdepth: %s", line);
            return 1;
        }
        *sdepth = atof(substr);
    }
    else *sdepth = NULLVAL;
/*
 *  Chars 84-87: ndef
 */
    if (partline(substr, line, 83, 4)) {
        if (check_int(substr)) {
            sprintf(isf_error, "bad ndef: %s", line);
            return 1;
        }
        *ndef = atoi(substr);
    }
    else *ndef = NULLVAL;
/*
 *  Chars 89-92: nsta
 */
    if (partline(substr, line, 88, 4)) {
        if (check_int(substr)) {
            sprintf(isf_error, "bad nsta: %s", line);
            return 1;
        }
        *nsta = atoi(substr);
    }
    else *nsta = NULLVAL;
/*
 *  Chars 94-96: gap
 */
    if (partline(substr, line, 93, 3)) {
        if (check_int(substr)) {
            sprintf(isf_error, "bad gap: %s", line);
            return 1;
        }
        *gap = atoi(substr);
    }
    else *gap = NULLVAL;
/*
 *  Chars 98-103: minimum distance
 */
    if (partline(substr, line, 97, 6)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad mindist: %s", line);
            return 1;
        }
        *mindist = atof(substr);
    }
    else *mindist = NULLVAL;
/*
 *  Chars 105-110: maximum distance
 */
    if (partline(substr, line, 104, 6)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad maxdist: %s", line);
            return 1;
        }
        *maxdist = atof(substr);
    }
    else *maxdist = NULLVAL;
/*
 *  Chars 116-117: event type
 */
    if (!partline(etype, line, 115, 2))
        strcpy(etype, "");
    else if (strlen(etype) != 2) {
        sprintf(isf_error, "bad etype: %s", line);
        return 1;
    }
/*
 *  Chars 119-127: author
 */
    if (!partline(author, line, 118, 9)) {
        sprintf(isf_error, "missing author: %s", line);
        return 1;
    }
    if (check_whole(author)) {
        sprintf(isf_error, "bad author: %s", line);
        return 1;
    }
/*
 *  Chars 129-139: origin ID
 */
    *hypid = NULLVAL;
    if (partline(origid, line, 128, 11)) {
        if (!check_int(origid))
            *hypid = atoi(origid);
    }
    else {
        sprintf(isf_error, "missing origid: %s", line);
        return 1;
    }
/*
 *  Chars 141-145: reporter
 */
    if (partline(hrep, line, 140, 5)) {
        if (check_whole(hrep)) {
            sprintf(isf_error, "bad reporter: %s", line);
            return 1;
        }
    }
    else strcpy(hrep, "");
/*
 *  Chars 147-151: depth-phase depth
 */
    *depdp = NULLVAL;
    if (partline(substr, line, 146, 5)) {
        if (!check_float(substr))
            *depdp = atof(substr);
    }
/*
 *  Chars 153-157: depth-phase depth error
 */
    *dpderr = NULLVAL;
    if (partline(substr, line, 152, 5)) {
        if (!check_float(substr))
            *dpderr = atof(substr);
    }
/*
 *  Chars 159-161: sgap
 */
    *sgap = NULLVAL;
    if (partline(substr, line, 158, 3)) {
        if (!check_int(substr))
            *sgap = atoi(substr);
    }
    return 0;
}

/*
 *  Parses a phase block data line.
 *     Only those values are parsed that are used by the locator.
 *     Returns 0/1 on success/error
 */
static int parse_phase(char *line, char *sta, double *dist, char *phase,
                       int *hh, int *mi, int *ss, int *msec, char *arrid,
                       double *azim, double *slow, double *amp, double *per,
                       char *sp_fm, char *detchar, char *agy, char *deploy,
                       char *lcn, char *auth, char *rep, char *pch, char *ach,
                       char *lp_fm, double *snr)
{
    char substr[LINLEN], c;
/*
 *  Chars 1-5: station code.
 */
    if (!partline(sta, line, 0, 5)) {
        sprintf(isf_error, "missing sta: %s", line);
        return 1;
    }
    if (check_whole(sta)) {
        sprintf(isf_error, "bad sta: %s", line);
        return 1;
    }
/*
 *  Chars 7-12: distance
 */
    if (partline(substr, line, 6, 6)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad distance: %s", line);
            return 1;
        }
        *dist = atof(substr);
    }
    else *dist = NULLVAL;
/*
 *  Chars 20-27: phase code - can be null.
 */
    if (partline(phase, line, 19, 8)) {
        if (check_whole(phase)) {
            sprintf(isf_error, "bad phase: %s", line);
            return 1;
        }
    }
    else strcpy(phase, "");
/*
 *  Chars 29-40: arrival time - can be null.
 */
    if (partline(substr, line, 28, 12)) {
/*
 *      Chars 29,30: hour.
 */
        if (!partline(substr, line, 28, 2)) {
            sprintf(isf_error, "missing hour: %s", line);
            return 1;
        }
        if (check_int(substr)) {
            sprintf(isf_error, "bad hour: %s", line);
            return 1;
        }
        *hh = atoi(substr);
/*
 *      Chars 32,33: minute
 */
        if (!partline(substr, line, 31, 2)) {
            sprintf(isf_error, "missing minute: %s", line);
            return 1;
        }
        if (check_int(substr)) {
            sprintf(isf_error, "bad minute: %s", line);
            return 1;
        }
        *mi = atoi(substr);
/*
 *      Chars 35,36: second.
 */
        if (!partline(substr, line, 34, 2)) {
            sprintf(isf_error, "missing second: %s", line);
            return 1;
        }
        if (check_int(substr)) {
            sprintf(isf_error, "bad second: %s", line);
            return 1;
        }
        *ss = atoi(substr);
/*
 *      Char 37-40: msec or spaces.
 *      Allow decimal place without any numbers after it.
 */
        if (partline(substr, line, 37, 3)) {
/*
 *          Char 37: '.' character
 */
            if (line[36] != '.') {
                sprintf(isf_error, "bad time: %s", line);
                return 1;
            }
/*
 *          Chars 38-40: msec.
 */
            if (!isdigit(line[37])) {
                sprintf(isf_error, "bad time: %s", line);
                return 1;
            }
            *msec = (line[37] - '0') * 100;
            if (isdigit(line[38]))
                *msec += (line[38] - '0') * 10;
            else if (line[38] != ' ' || line[39] != ' ') {
                sprintf(isf_error, "bad time: %s", line);
                return 1;
            }
            if (isdigit(line[39]))
                *msec += (line[39] - '0');
            else if (line[39] != ' ') {
                sprintf(isf_error, "bad time: %s", line);
                return 1;
            }
        }
        else {
/*
 *          Char 37: '.' character or space
 */
            if (line[36] != '.' && line[36] != ' ') {
                sprintf(isf_error, "bad time: %s", line);
                return 1;
            }
            *msec = NULLVAL;
        }
    }
    else {
        *hh = NULLVAL;
        *mi = NULLVAL;
        *ss = NULLVAL;
        *msec = NULLVAL;
    }
/*
 *  Chars 48-52: observed azimuth
 */
    if (partline(substr, line, 47, 5)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad azim: %s", line);
            return 1;
        }
        *azim = atof(substr);
    }
    else *azim = NULLVAL;
/*
 *  Chars 60-65: slowness
 */
    if (partline(substr, line, 59, 6)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad slow: %s", line);
            return 1;
        }
        *slow = atof(substr);
    }
    else *slow = NULLVAL;
/*
 *  Chars 78-82: snr
 */
    if (partline(substr, line, 77, 5)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad snr: %s", line);
            return 1;
        }
        *snr = atof(substr);
    }
    else *snr = NULLVAL;
/*
 *  Chars 84-92: amplitude
 */
    if (partline(substr, line, 83, 9)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad amp: %s", line);
            return 1;
        }
        *amp = atof(substr);
    }
    else *amp = NULLVAL;
/*
 *  Chars 94-98: period
 */
    if (partline(substr, line, 93, 5)) {
        if (check_float(substr)) {
            sprintf(isf_error, "bad per: %s", line);
            return 1;
        }
        *per = atof(substr);
    }
    else *per = NULLVAL;
/*
 *  Char 101: sp_fm
 */
    c = tolower(line[100]);
    if (c == 'c' || c == 'd') *sp_fm = c;
    else                      *sp_fm = '_';
/*
 *  Char 102: detection character
 */
    c = tolower(line[101]);
    if (c == 'i' || c == 'e' || c == 'q') *detchar = c;
    else                                  *detchar = '_';
/*
 *  ISF2 extra columns
 */
/*
 *  Chars 115-125: arrival id
 */
    if (partline(arrid, line, 114, 11)) {
        if (check_whole(arrid)) {
            sprintf(isf_error, "bad agency: %s", line);
            return 1;
        }
    }
    else strcpy(arrid, "");
/*
 *  Chars 127-131: agency code
 */
    if (partline(agy, line, 126, 5)) {
        if (check_whole(agy)) {
            sprintf(isf_error, "bad agency: %s", line);
            return 1;
        }
    }
    else strcpy(agy, "FDSN");
/*
 *  Chars 133-140: deployment code
 */
    if (partline(deploy, line, 132, 8)) {
        if (check_whole(deploy)) {
            sprintf(isf_error, "bad deployment: %s", line);
            return 1;
        }
    }
    else strcpy(deploy, "--");
/*
 *  Chars 142-143: location code
 */
    if (partline(lcn, line, 141, 2)) {
        if (check_whole(lcn)) {
            sprintf(isf_error, "bad location: %s", line);
            return 1;
        }
    }
    else strcpy(lcn, "--");
/*
 *  Chars 145-149: author code
 */
    if (partline(auth, line, 144, 5)) {
        if (check_whole(auth)) {
            sprintf(isf_error, "bad author: %s", line);
            return 1;
        }
    }
    else strcpy(auth, "");
/*
 *  Chars 151-155: reporter code
 */
    if (partline(rep, line, 150, 5)) {
        if (check_whole(rep)) {
            sprintf(isf_error, "bad reporter: %s", line);
            return 1;
        }
    }
    else strcpy(rep, "");
/*
 *  Chars 157-159: pick channel code
 */
    if (partline(pch, line, 156, 3)) {
        if (check_whole(pch)) {
            sprintf(isf_error, "bad phase channel: %s", line);
            return 1;
        }
    }
    else strcpy(pch, "???");
/*
 *  Chars 157-159: amplitude channel code
 */
    if (partline(ach, line, 160, 3)) {
        if (check_whole(ach)) {
            sprintf(isf_error, "bad amplitude channel: %s", line);
            return 1;
        }
    }
    else strcpy(ach, "???");
/*
 *  Char 165: long period first motion
 */
    c = tolower(line[164]);
    if (c == 'd' || c == 'c') *lp_fm = c;
    else                      *lp_fm = '_';
    return 0;
}

/*
 * Title:
 *    write_isf
 * Synopsis:
 *    Writes solution to ISF bulletin format file.
 * Input Arguments:
 *    ofp       - file pointer to write to.
 *    ep        - pointer to event info
 *    sp        - pointer to current solution.
 *    p[]       - array of phases.
 *    h[]       - array of hypocentres.
 *    stamag_mb - array of mb stamag structures
 *    stamag_ms - array of Ms stamag structures
 *    rdmag_mb  - pointer to mb reading mag structures
 *    rdmag_ms  - pointer to MS reading mag structures
 *    grn       - geographic region number
 *    magbloc   - reported magnitudes from ISF input file
 *  Called by:
 *     main
 * Calls:
 *    gregion, human_time_isf, write_origin, write_phase
 */
void write_isf(FILE *ofp, EVREC *ep, SOLREC *sp, HYPREC h[], PHAREC p[],
              STAMAG *stamag_mb, STAMAG *stamag_ms,
              RDMAG rdmag_mb[], RDMAG rdmag_ms[], int grn, char *magbloc)
{
    extern char out_agency[];                       /* From config file */
    extern char ms_phase[MAXNUMPHA][PHALEN];         /* From model file */
    extern char mb_phase[MAXNUMPHA][PHALEN];         /* From model file */
    extern int mb_phase_num;                         /* From model file */
    extern int ms_phase_num;                         /* From model file */
    char antype, loctype, epifix, depfix, timfix;
    char hday[25], htim[25], gregname[255], phase[PHALEN];
    char timedef, id[15], prevsta[STALEN], magtype[6];
    double amp = 0., per = 0., mag = 0., x = NULLVAL;
    int i, j, rdid, ismb, isms;
    antype = loctype = ' ';
/*
 *  Write event header
 */
    fprintf(ofp, "BEGIN IMS2.0\n");
    fprintf(ofp, "DATA_TYPE BULLETIN IMS1.0:short with ISF2.0 extensions\n");
    strcpy(gregname, "");
    if (grn) gregion(grn, gregname);
    fprintf(ofp, "\nEvent %s %-s\n\n", ep->eventid, gregname);
/*
 *  Write origin header
 */
    fprintf(ofp, "   Date       Time        Err   RMS Latitude Longitude  ");
    fprintf(ofp, "Smaj  Smin  Az Depth   Err Ndef Nsta Gap  mdist  Mdist ");
    fprintf(ofp, "Qual   Author      OrigID    Rep   DPdep   Err Sgp\n");
/*
 *  Write old hypocentres
 */
    for (i = 0; i < ep->numhyps; i++) {
/*
 *      skip old prime if it was out_agency
 */
        if (streq(h[i].agency, out_agency)) continue;
        human_time_isf(hday, htim, h[i].time);
        htim[11] = '\0';
        depfix = epifix = timfix = ' ';
        if (h[i].depfix) depfix = 'f';
        if (h[i].epifix) epifix = 'f';
        if (h[i].timfix) timfix = 'f';
        write_origin(ofp, hday, htim, timfix, h[i].stime, h[i].sdobs,
                     h[i].lat, h[i].lon, epifix,
                     h[i].smajax, h[i].sminax, h[i].strike,
                     h[i].depth, depfix, h[i].sdepth, h[i].ndef, h[i].ndefsta,
                     h[i].azimgap, h[i].mindist, h[i].maxdist, antype,
                     loctype, h[i].etype, h[i].agency, h[i].origid, h[i].rep,
                     h[i].depdp, h[i].dpderr, h[i].sgap);
    }
/*
 *  Write new solution.
 */
    human_time_isf(hday, htim, sp->time);
    htim[11] = '\0';
    depfix = epifix = timfix = ' ';
    if (sp->depfix) depfix = 'f';
    if (ep->fix_depth_depdp) depfix = 'd';
    if (sp->epifix) epifix = 'f';
    if (sp->timfix) timfix = 'f';
    sprintf(id, "%d", sp->hypid);
    write_origin(ofp, hday, htim, timfix, sp->error[0], sp->sdobs,
                 sp->lat, sp->lon, epifix,
                 sp->smajax, sp->sminax, sp->strike,
                 sp->depth, depfix, sp->error[3], sp->ndef, sp->ndefsta,
                 sp->azimgap, sp->mindist, sp->maxdist, antype,
                 loctype, ep->etype, out_agency, id, out_agency,
                 sp->depdp, sp->depdp_error, sp->sgap);
    fprintf(ofp, " (#PRIME)\n\n");
/*
 *  Write magnitude block
 */
    i = strlen(magbloc);
    if (i > 0 || sp->bodymag != NULLVAL || sp->surfmag != NULLVAL)
        fprintf(ofp, "Magnitude  Err Nsta Author      OrigID\n");
    if (i > 0) fprintf(ofp, "%s", magbloc);
    if (sp->bodymag != NULLVAL)
        fprintf(ofp, "mb    %4.1f %3.1f %4d %-9s %d\n",
                sp->bodymag, sp->bodymag_uncertainty, sp->nsta_mb,
                out_agency, sp->hypid);
    if (sp->surfmag != NULLVAL)
        fprintf(ofp, "MS    %4.1f %3.1f %4d %-9s %d\n",
                sp->surfmag, sp->surfmag_uncertainty, sp->nsta_ms,
                out_agency, sp->hypid);
/*
 *  Write phase block
 */
    fprintf(ofp, "\nSta     Dist  EvAz Phase        Time      TRes  ");
    fprintf(ofp, "Azim AzRes   Slow   SRes Def   SNR       Amp   Per ");
    fprintf(ofp, "Qual Magnitude    ArrID    Agy   Deploy   Ln Auth  ");
    fprintf(ofp, "Rep   PCh ACh L\n");
    rdid = -1;
    strcpy(prevsta, "");
    for (i = 0; i < ep->numphas; i++) {
/*
 *      comment lines for reading magnitudes
 */
        if (p[i].rdid != rdid && i) {
            for (j = 0; j < sp->nass_mb; j++) {
                if (rdid != rdmag_mb[j].rdid)
                    continue;
                fprintf(ofp, " (Reading mb: %4.1f magdef=%d\n",
                        rdmag_mb[j].magnitude, rdmag_mb[j].magdef);
            }
            for (j = 0; j < sp->nass_ms; j++) {
                if (rdid != rdmag_ms[j].rdid)
                    continue;
                fprintf(ofp, " (Reading MS: %4.1f magdef=%d\n",
                        rdmag_ms[j].magnitude, rdmag_ms[j].magdef);
            }
        }
        rdid = p[i].rdid;
/*
 *      comment lines for station magnitudes
 */
        if (strcmp(p[i].prista, prevsta) && i) {
            for (j = 0; j < sp->nsta_mb; j++) {
                if (streq(stamag_mb[j].sta, prevsta)) {
                    fprintf(ofp, " (Station mb: %4.1f magdef=%d\n",
                            stamag_mb[j].magnitude, stamag_mb[j].magdef);
                    break;
                }
            }
            for (j = 0; j < sp->nsta_ms; j++) {
                if (streq(stamag_ms[j].sta, prevsta)) {
                    fprintf(ofp, " (Station MS: %4.1f magdef=%d\n",
                            stamag_ms[j].magnitude, stamag_ms[j].magdef);
                    break;
                }
            }
        }
        strcpy(prevsta, p[i].prista);
/*
 *      phase lines
 */
        amp = per = NULLVAL;
        human_time_isf(hday, htim, p[i].time);
        if (p[i].timedef) timedef = 'T';
        else              timedef = '_';
        mag = NULLVAL;
        sprintf(id, "???");
        strcpy(magtype, "     ");
        if (p[i].numamps > 0) {
            amp = p[i].a[0].amp;
            per = p[i].a[0].per;
            if (p[i].a[0].mtypeid) mag = p[i].a[0].magnitude;
            if (p[i].a[0].mtypeid == 1) strcpy(magtype, "mb");
            if (p[i].a[0].mtypeid == 2) strcpy(magtype, "MS");
            sprintf(id, "%s", p[i].a[0].ach);
        }
        write_phase(ofp, p[i].sta, p[i].delta, p[i].esaz, p[i].phase,
                    htim, p[i].resid, p[i].azim, p[i].slow, timedef,
                    p[i].snr, amp, per, p[i].sp_fm, p[i].detchar, magtype,
                    mag, p[i].arrid, "FDSN", p[i].deploy, p[i].lcn,
                    p[i].auth, p[i].rep, p[i].pch, id, p[i].lp_fm);
/*
 *      fake phase lines for extra amplitudes, periods
 */
        ismb = isms = 0;
        if (p[i].numamps > 1) {
            for (j = 0; j < mb_phase_num; j++)
                if (streq(p[i].phase, mb_phase[j]))
                    ismb = 1;
            for (j = 0; j < ms_phase_num; j++)
                if (streq(p[i].phase, ms_phase[j]))
                    isms = 1;
            if (ismb) strcpy(phase, "AMB");
            if (isms) strcpy(phase, "AMS");
            if (ismb || isms) {
                strcpy(htim, "            ");
                for (j = 1; j < p[i].numamps; j++)
                    mag = NULLVAL;
                    strcpy(magtype, "     ");
                    if (p[i].a[0].mtypeid) mag = p[i].a[0].magnitude;
                    if (p[i].a[0].mtypeid == 1) strcpy(magtype, "mb");
                    if (p[i].a[0].mtypeid == 2) strcpy(magtype, "MS");
                    write_phase(ofp, p[i].sta, p[i].delta, p[i].esaz, phase,
                                htim, x, x, x, '_', x,
                                p[i].a[j].amp, p[i].a[j].per, 'x', 'x',
                                magtype, mag, p[i].arrid, "FDSN",
                                p[i].deploy, p[i].lcn, p[i].auth, p[i].rep,
                                "???", p[i].a[j].ach, '_');
            }
        }
    }
/*
 *  last reading
 */
    for (j = 0; j < sp->nass_mb; j++) {
        if (rdid != rdmag_mb[j].rdid)
            continue;
        fprintf(ofp, " (Reading mb: %4.1f magdef=%d\n",
               rdmag_mb[j].magnitude, rdmag_mb[j].magdef);
    }
    for (j = 0; j < sp->nass_ms; j++) {
        if (rdid != rdmag_ms[j].rdid)
            continue;
        fprintf(ofp, " (Reading MS: %4.1f magdef=%d\n",
                rdmag_ms[j].magnitude, rdmag_ms[j].magdef);
    }
/*
 *  last station
 */
    for (j = 0; j < sp->nsta_mb; j++) {
        if (streq(stamag_mb[j].sta, prevsta)) {
            fprintf(ofp, " (Station mb: %4.1f magdef=%d\n",
                    stamag_mb[j].magnitude, stamag_mb[j].magdef);
            break;
        }
    }
    for (j = 0; j < sp->nsta_ms; j++) {
        if (streq(stamag_ms[j].sta, prevsta)) {
            fprintf(ofp, " (Station MS: %4.1f magdef=%d\n",
                    stamag_ms[j].magnitude, stamag_ms[j].magdef);
            break;
        }
    }
    fprintf(ofp, "\nSTOP\n");
}

/*
 *  Writes an origin line
 */
static void write_origin(FILE *fp, char *hday, char *htim, char timfix,
                         double stime, double sdobs, double lat, double lon,
                         char epifix, double smaj, double smin, int strike,
                         double depth, char depfix, double sdepth, int ndef,
                         int nsta, int gap, double mindist, double maxdist,
                         char antype, char loctype, char *etype,
                         char *author, char *origid, char *reporter,
                         double depdp, double dpderr, int sgap)
{
/*
 *  Chars 1-24: origin time and fixed time flag
 */
    fprintf(fp, "%s %s%c ", hday, htim, timfix);
/*
 *  Chars 25-30: origin time error
 */
    if (stime == NULLVAL)   fprintf(fp, "      ");
    else if (stime > 99.99) fprintf(fp, "99.99 ");
    else                    fprintf(fp, "%5.2f ", stime);
/*
 *  Chars 31-36: sdobs
 */
    if (sdobs == NULLVAL)   fprintf(fp, "      ");
    else if (sdobs > 99.99) fprintf(fp, "99.99 ");
    else                    fprintf(fp, "%5.2f ", sdobs);
/*
 *  Chars 37-55: latitude, longitude and fixed epicentre flag
 */
    fprintf(fp, "%8.4f %9.4f%c", lat, lon, epifix);
/*
 *  Chars 56-71: optional semi-major, semi-minor axes, strike
 */
    if (smaj == NULLVAL)   fprintf(fp, "      ");
    else if (smaj > 999.9) fprintf(fp, "999.9 ");
    else                   fprintf(fp, "%5.1f ", smaj);
    if (smin == NULLVAL)   fprintf(fp, "      ");
    else if (smin > 999.9) fprintf(fp, "999.9 ");
    else                   fprintf(fp, "%5.1f ", smin);
    if (strike == NULLVAL) fprintf(fp, "    ");
    else                   fprintf(fp, "%3d ", strike);
/*
 *  Chars 72-78: optional depth and fixed depth flag
 */
    if (depth == NULLVAL)  fprintf(fp, "     %c ", depfix);
    else                   fprintf(fp, "%5.1f%c ", depth, depfix);
/*
 *  Chars 79-83: optional depth error
 */
    if (sdepth == NULLVAL) fprintf(fp, "     ");
    else                   fprintf(fp, "%4.1f ", sdepth);
/*
 *  Chars 84-111: optional ndef, nsta, gap, mindist, maxdist
 */
    if (ndef == NULLVAL)    fprintf(fp, "     ");
    else                    fprintf(fp, "%4d ", ndef);
    if (nsta == NULLVAL)    fprintf(fp, "     ");
    else                    fprintf(fp, "%4d ", nsta);
    if (gap == NULLVAL)     fprintf(fp, "    ");
    else                    fprintf(fp, "%3d ", gap);
    if (mindist == NULLVAL) fprintf(fp, "       ");
    else                    fprintf(fp, "%6.2f ", mindist);
    if (maxdist == NULLVAL) fprintf(fp, "       ");
    else                    fprintf(fp, "%6.2f ", maxdist);
/*
 *  Char 112-115: analysis type, location method
 */
    fprintf(fp, "%c %c ", antype, loctype);
/*
 *  Chars 116-118: event type
 */
    fprintf(fp, "%-2s ", etype);
/*
 *  Chars 119-128: author
 */
    fprintf(fp, "%-9s ", author);
/*
 *  Chars 129-139: origid
 */
    fprintf(fp, "%-11s ", origid);
/*
 *  Chars 141-145: reporter
 */
    fprintf(fp, "%-5s ", reporter);
/*
 *  Chars 147-161: depth-phase depth, depdp error, sgap
 */
    if (depdp == NULLVAL)  fprintf(fp, "      ");
    else                   fprintf(fp, "%5.1f ", depdp);
    if (dpderr == NULLVAL) fprintf(fp, "      ");
    else                   fprintf(fp, "%5.1f ", dpderr);
    if (sgap == NULLVAL)   fprintf(fp, "    ");
    else                   fprintf(fp, "%3d ", sgap);
    fprintf(fp, "\n");
}

/*
 * Writes a phase block data line.
 */
static void write_phase(FILE *fp, char *sta, double dist, double esaz,
                        char *phase, char *htim, double timeres,
                        double azim, double slow, char timedef, double snr,
                        double amp, double per, char sp_fm, char detchar,
                        char *magtype, double mag, char *arrid,
                        char *agency, char *deploy, char *lcn, char *auth,
                        char *rep, char *pch, char *ach, char lp_fm)
{
    char c, d;
/*
 *  Chars 1-5: station code
 */
    fprintf(fp, "%-5s ", sta);
/*
 *  Chars 7-12: distance
 */
    if (dist == NULLVAL)  fprintf(fp, "       ");
    else                  fprintf(fp, "%6.2f ", dist);
/*
 *  Chars 14-18: event to sta azimuth
 */
    if (esaz == NULLVAL) fprintf(fp, "      ");
    else                 fprintf(fp, "%5.1f ", esaz);
/*
 *  Chars 20-27: phase code - can be null
 */
    fprintf(fp, "%-8s ", phase);
/*
 *  Chars 29-40: time. Time can be completely null.
 */
    fprintf(fp, "%s ", htim);
/*
 *  Chars 42-46: time residual
 */
    if (timeres == NULLVAL)  fprintf(fp, "      ");
    else {
        if (timeres >  9999) timeres =  9999;
        if (timeres < -9999) timeres = -9999;
        fprintf(fp, "%5.1f ", timeres);
    }
/*
 *  Chars 48-58: observed azimuth and azimuth residual
 */
    if (azim == NULLVAL) fprintf(fp, "            ");
    else                 fprintf(fp, "%5.1f       ", azim);
/*
 *  Chars 60-72: slownessand slowness residual
 */
    if (slow == NULLVAL) fprintf(fp, "              ");
    else                 fprintf(fp, "%6.2f        ", slow);
/*
 *  Char 74-76: defining flags
 */
    fprintf(fp, "%c__ ", timedef);
/*
 *  Chars 78-82: signal-to noise ratio
 */
    if (snr == NULLVAL)  fprintf(fp, "      ");
    else                 fprintf(fp, "%5.1f ", snr);
/*
 *  Chars 84-92: amplitude
 */
    if (amp == NULLVAL) fprintf(fp, "          ");
    else                fprintf(fp, "%9.1f ", amp);
/*
 *  Chars 94-98: period
 */
    if (per == NULLVAL) fprintf(fp, "      ");
    else                fprintf(fp, "%5.2f ", per);
/*
 *  Char 100-102: picktype, sp_fm, detchar
 */
    d = tolower(sp_fm);
    if (d == 'c' || d == 'u' || d == '+') c = 'c';
    else if (d == 'd' || d == '-')        c = 'd';
    else                                  c = '_';
    d = tolower(detchar);
    if (!(d == 'e' || d == 'i' || d == 'q')) d = '_';
    fprintf(fp, "_%c%c ", c, d);
/*
 *  Chars 104-109: magnitude type, magnitude indicator
 */
    fprintf(fp, "%-5s ", magtype);
/*
 *  Chars 110-113: magnitude
 */
    if (mag == NULLVAL) fprintf(fp, "     ");
    else                fprintf(fp, "%4.1f ", mag);
/*
 *  Chars 115-125: arrival ID
 */
    fprintf(fp, "%-11s ", arrid);
/*
 *  ISF2 extensions
 */
    fprintf(fp, "%-5s %-8s %-2s ", agency, deploy, lcn);
    fprintf(fp, "%-5s %-5s ", auth, rep);
    fprintf(fp, "%3s %3s ", pch, ach);
    d = tolower(lp_fm);
    if (d == 'c' || d == 'u' || d == '+') c = 'c';
    else if (d == 'd' || d == '-')        c = 'd';
    else                                  c = '_';
    fprintf(fp, "%c\n", d);
}


/*
 *  Get a substring, removing leading and trailing white space.
 *  Expects a string, an offset from the start of the string, and a maximum
 *  length for the resulting substring. If this length is 0 it will take up
 *  to the end of the input string.
 *
 *  Need to allow for ')' to come after the required field at the end of a
 *  comment.  Discard ')' at end of a string  as long as there's no '('
 *  before it anywhere.
 *
 *  Returns the length of the resulting substring.
 */
static int partline(char *part, char *line, int offset, int numchars)
{
    int i, j, len;
    int bracket = 0;
    len = (int)strlen(line);
    if (len < offset) return 0;
    if (numchars == 0) numchars = len - offset;
    for (i = offset, j = 0; i < offset + numchars; i++) {
        if (j == 0 && (line[i] == ' ' || line[i] == '\t')) continue;
        if (line[i] == '\0' || line[i] == '\n') break;
        part[j++] = line[i];
        if (line[i] == '(') bracket = 1;
    }
    if (!bracket) {
        while (--j != -1 && (part[j] == ' ' ||
                             part[j] == '\t' || part[j] == ')'));
        part[++j] = '\0';
    }
    else if (j) {
        while (part[--j] == ' ' || part[j] == '\t');
        part[++j] = '\0';
    }
    return j;
}

/*
 *  Check that a string has no spaces in it.
 *  Returns 0 if there are no spaces or 1 if there is a space.
 */
static int check_whole(char *substr)
{
    int i, n;
    n = strlen(substr);
    for (i = 0; i < n; i++) {
        if (substr[i] == ' ' || substr[i] == '\t')
            return 1;
    }
    return 0;
}

/*
 *  Check if a string is composed entirely of white space or not.
 *  Returns 1 if it is, 0 if it isn't.
 */
static int all_blank(char *s)
{
    int i;
    for (i = 0; s[i] == ' ' || s[i] == '\t'; i++);
    if (s[i] == '\n' || s[i] == '\0') return 1;
    return 0;
}

/*
 *  Check that a string contains only sign/number characters
 *  atoi itself does no checking.
 *  Returns 0 if OK,  1 if not.
 */
int check_int(char *substr)
{
    int i, n;
    n = strlen(substr);
    for (i = 0; i < n; i++) {
        if (isdigit(substr[i])) continue;
        if (i == 0)
            if (substr[i] == '+' || substr[i] == '-') continue;
        return 1;
    }
    return 0;
}

/*
 *  Check that a string contains only sign/number/decimal point characters
 *  atof itself does no checking.
 *  Returns 0 if OK,  1 if not.
 */
static int check_float(char *substr)
{
    int i, n;
    n = strlen(substr);
    for (i=0; i < n; i++) {
        if (isdigit(substr[i]) || substr[i] == '.') continue;
        if (i == 0)
            if (substr[i] == '+' || substr[i] == '-') continue;
        return 1;
    }
    return 0;
}

/*
 *  Title:
 *     sort_phaserec_isf
 *  Synopsis:
 *     Sorts phase structures by increasing delta, prista, time.
 *  Input Arguments:
 *     numphas    - number of associated phases
 *     p[]        - array of phase structures.
 *  Called by:
 *     read_isf
 */
void sort_phaserec_isf(int numphas, PHAREC p[])
{
    int i, j;
    PHAREC temp;
    for (i = 1; i < numphas; i++) {
        for (j = i - 1; j > -1; j--) {
            if ((p[j].time > p[j+1].time && p[j+1].time != NULLVAL) ||
                 p[j].time == NULLVAL) {
                swap(p[j], p[j+1]);
            }
        }
    }
    for (i = 1; i < numphas; i++) {
        for (j = i - 1; j > -1; j--) {
            if (strcmp(p[j].prista, p[j+1].prista) > 0) {
                swap(p[j], p[j+1]);
            }
        }
    }
    for (i = 1; i < numphas; i++) {
        for (j = i - 1; j > -1; j--) {
            if (p[j].delta > p[j+1].delta) {
                swap(p[j], p[j+1]);
            }
        }
    }
}

/*  EOF  */
