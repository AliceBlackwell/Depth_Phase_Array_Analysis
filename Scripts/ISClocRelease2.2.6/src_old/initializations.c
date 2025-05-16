#include "iscloc.h"
extern int verbose;
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;

/*
 * Functions:
 *    init_event
 *    start_hyp
 *    init_sol
 */

/*
 *  Title:
 *     init_event
 *  Synopsis:
 *     Deals with depth_agency/location_agency/time_agency instructions.
 *     If event identified as anthropogenic then fix depth at surface,
 *        unless being fixed explicitly to some depth.
 *  Input Arguments:
 *     ep  - pointer to event info
 *     h[] - array of hypocentre structures
 *     p[] - array of phase structures
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     eventloc
 */
int init_event(EVREC *ep, HYPREC h[])
{
    extern double default_depth;    /* From config file */
    HYPREC temp;
    int manmade = 0;
    int i;
/*
 *  check for anthropogenic events
 */
    if (verbose) fprintf(logfp, "etype=%s\n", ep->etype);
    if (ep->etype[1] == 'n' ||
        ep->etype[1] == 'x' ||
        ep->etype[1] == 'm' ||
        ep->etype[1] == 'q' ||
        ep->etype[1] == 'r' ||
        ep->etype[1] == 'h' ||
        ep->etype[1] == 's' ||
        ep->etype[1] == 'i')
        manmade = 1;
/*
 *  If event identified as anthropogenic then fix depth at surface,
 *  unless being fixed explicitly to some depth.
 */
    if (manmade) {
        if (!ep->depth_fix) {
            ep->start_depth = 0;
            ep->depth_fix   = 1;
            ep->surface_fix = 1;    /* To stop options 2 and 3. */
            fprintf(logfp, "Fix depth to zero because etype=%s\n", ep->etype);
        }
        else {
            ep->surface_fix = 1;    /* To stop options 2 and 3. */
        }
    }
/*
 *  depth_agency instruction - set depth to that given by chosen agency.
 */
    if (ep->depth_agency[0] && ep->start_depth == NULLVAL) {
        for (i = 0; i < ep->numhyps; i++) {
            if (streq(h[i].agency, ep->depth_agency)) {
                ep->start_depth = h[i].depth;
                break;
            }
        }
        if (ep->start_depth == NULLVAL) {
            fprintf(errfp, "ABORT: %d invalid depth agency %s!\n",
                    ep->evid, ep->depth_agency);
            fprintf(logfp, "ABORT: %d invalid depth agency %s!\n",
                    ep->evid, ep->depth_agency);
            return 1;
        }
    }
/*
 *  location_agency instruction - set lat,lon to that given by agency,
 */
    if (ep->location_agency[0] && ep->start_lat == NULLVAL) {
        for (i = 0; i < ep->numhyps; i++) {
            if (streq(h[i].agency, ep->location_agency)) {
                ep->start_lat = h[i].lat;
                ep->start_lon = h[i].lon;
                break;
            }
        }
        if (ep->start_lat == NULLVAL) {
            fprintf(errfp, "ABORT: %d invalid location agency %s!\n",
                    ep->evid, ep->location_agency);
            fprintf(logfp, "ABORT: %d invalid location agency %s!\n",
                    ep->evid, ep->location_agency);
            return 1;
        }
    }
/*
 *  time_agency instruction - set time to that given by agency.
 */
    if (ep->time_agency[0] && ep->start_time == NULLVAL) {
        for (i = 0; i < ep->numhyps; i++) {
            if (streq(h[i].agency, ep->time_agency)) {
                ep->start_time = h[i].time;
                break;
            }
        }
        if (ep->start_time == NULLVAL) {
            fprintf(errfp, "ABORT: %d invalid time agency %s!\n",
                    ep->evid, ep->time_agency);
            fprintf(logfp, "ABORT: %d invalid time agency %s!\n",
                    ep->evid, ep->time_agency);
            return 1;
        }
    }
/*
 *  hypo_agency instruction - set hypocenter to that given by agency.
 */
    if (ep->hypo_agency[0]) {
        for (i = 0; i < ep->numhyps; i++) {
            if (streq(h[i].agency, ep->hypo_agency)) {
                ep->start_time  = h[i].time;
                ep->start_lat   = h[i].lat;
                ep->start_lon   = h[i].lon;
                ep->start_depth = h[i].depth;
                swap(h[i], h[0]);
                break;
            }
        }
        if (ep->start_time == NULLVAL) {
            fprintf(errfp, "ABORT: %d invalid hypo agency %s!\n",
                    ep->evid, ep->hypo_agency);
            fprintf(logfp, "ABORT: %d invalid hypo agency %s!\n",
                    ep->evid, ep->hypo_agency);
            return 1;
        }
        ep->hypo_fix = 1;
    }
/*
 *  Set hypo_fix if everything is fixed separately
 *      known issue: prime_hypid is undefined!
 */
    if (ep->epi_fix && ep->depth_fix && ep->time_fix)
        ep->hypo_fix = 1;
    if (ep->start_depth != NULLVAL && ep->start_depth < 0.)
        ep->start_depth = default_depth;
    return 0;
}

/*
 *  Title:
 *     start_hyp
 *  Synopsis:
 *     Sets starting hypocentre to median of all hypocentre parameters
 *        if prime is not ISC or IASPEI or EHB
 *  Input Arguments:
 *     ep  - pointer to event info
 *     h[] - array of hypocentre structures
 *  Output Arguments:
 *     starthyp - pointer to starting hypocentre
 *  Called by:
 *     eventloc
 */
void start_hyp(EVREC *ep, HYPREC h[], HYPREC *starthyp)
{
    extern double default_depth;
    extern char nohypoagency[MAXBUF][AGLEN];
    extern int numnohypoagency;
    double *x = (double *)NULL;
    double medtim = 0., medlat = 0., medlon = 0., meddep = 0., z = 0., y;
    int i, j, n = 0, m;
    starthyp->time  = h[0].time;
    starthyp->lat   = h[0].lat;
    starthyp->lon   = h[0].lon;
    if (h[0].depth == NULLVAL || h[0].depth < 0.)
        starthyp->depth = default_depth;
    else
        starthyp->depth = h[0].depth;
    z = starthyp->depth;
/*
 *  single prime
 */
    if (ep->numhyps == 1) return;
/*
 *  flag hypocenters that may not be used in setting initial hypocentre
 */
    for (i = 0; i < ep->numhyps; i++) {
        h[i].ignorehypo = 0;
        for (j = 0; j < numnohypoagency; j++) {
            if (streq(h[i].agency, nohypoagency[j])) {
                h[i].ignorehypo = 1;
                break;
            }
        }
        if (!h[i].ignorehypo) n++;
    }
    m = n / 2;
    if ((x = (double *)calloc(n, sizeof(double))) == NULL) {
        fprintf(logfp, "start_hyp: cannot allocate memory!\n");
        fprintf(errfp, "start_hyp: cannot allocate memory!\n");
        errorcode = 1;
        return;
    }
/*
 *  calculate median hypocentre parameters
 */
    for (j = 0, i = 0; i < ep->numhyps; i++)
        if (!h[i].ignorehypo) x[j++] = h[i].time;
    if (n == 2)
        medtim = (x[0] + x[1]) / 2.;
    else {
        qsort(x, n, sizeof(double), double_compare);
        medtim = x[m];
    }

    for (j = 0, i = 0; i < ep->numhyps; i++) {
        if (!h[i].ignorehypo) x[j++] = h[i].lat;
    }
    if (n == 2)
        medlat = (x[0] + x[1]) / 2.;
    else {
        qsort(x, n, sizeof(double), double_compare);
        medlat = x[m];
    }

    for (j = 0, i = 0; i < ep->numhyps; i++) {
        if (!h[i].ignorehypo) x[j++] = h[i].lon;
    }
    if (n == 2) {
        y = fabs(x[0]) + fabs(x[1]);
        if (y > 180. && x[0] * x[1] < 0.) medlon = y / 2.;
        else                              medlon = (x[0] + x[1]) / 2.;
    }
    else {
        qsort(x, n, sizeof(double), double_compare);
        medlon = x[m];
    }

    for (j = 0, i = 0; i < ep->numhyps; i++) {
        if (!h[i].ignorehypo && h[i].depth != NULLVAL && h[i].depth >= 0.)
            x[j++] = h[i].depth;
    }
    if (j == 0)      meddep = default_depth;
    else if (j == 1) meddep = x[0];
    else if (j == 2) meddep = (x[0] + x[1]) / 2.;
    else {
        qsort(x, j, sizeof(double), double_compare);
        j /= 2;
        meddep = x[j];
    }
    Free(x);
    starthyp->time = medtim;
    starthyp->lat = medlat;
    starthyp->lon = medlon;
    starthyp->depth = meddep;
/*
 *  trusted primes: set initial epicentre/hypocentre to that
 */
    if (streq(h[0].agency, "IASPEI") ||
        streq(h[0].agency, "ISC")  ||
        streq(h[0].agency, "EHB")) {
        if (!ep->time_fix)
            starthyp->time = h[0].time;
        if (!ep->epi_fix) {
            starthyp->lat = h[0].lat;
            starthyp->lon = h[0].lon;
        }
        if (!ep->depth_fix)
            starthyp->depth = z;
    }
}


/*
 *  Title:
 *     init_sol
 *  Synopsis:
 *     Initializes solution structure with initial hypocentre guess.
 *  Input Arguments:
 *     sp  - pointer to solution structure to be initialised.
 *     ep  - pointer to event info
 *     hp  - pointer to starting hypocentre
 *  Return:
 *     0/1 on success/error.
 *  Called by:
 *     eventloc, fixedhypo
 */
int init_sol(SOLREC *sp, EVREC *ep, HYPREC *hp)
{
    extern double default_depth;                        /* From config file */
    int i, j;
/*
 *  Mark as not converged yet.
 */
    sp->converged = 0;
    sp->diverging = 0;
/*
 *  Check if there are sufficient number of observations
 */
    sp->numphas = ep->numphas;
    sp->nreading = ep->numrd;
    if (sp->numphas <= sp->number_of_unknowns) {
        fprintf(errfp, "init_sol: insufficient number of observations!\n");
        fprintf(errfp, "          unknowns=%d numphas=%d\n",
                sp->number_of_unknowns, sp->numphas);
        fprintf(logfp, "init_sol: insufficient number of observations!\n");
        fprintf(logfp, "          unknowns=%d numphas=%d\n",
                sp->number_of_unknowns, sp->numphas);
        errorcode = 5;
        return 1;
    }
/*
 *  Set starting point for location to the median hypocentre
 *  or to that set by the user.
 */
    sp->lat   = (ep->start_lat   != NULLVAL) ? ep->start_lat   : hp->lat;
    sp->lon   = (ep->start_lon   != NULLVAL) ? ep->start_lon   : hp->lon;
    sp->depth = (ep->start_depth != NULLVAL) ? ep->start_depth : hp->depth;
    sp->time  = (ep->start_time  != NULLVAL) ? ep->start_time  : hp->time;
/*
 *  set epifix and timfix flags
 */
    sp->epifix = ep->epi_fix;
    sp->timfix = ep->time_fix;
    sp->depfix = ep->depth_fix;
/*
 *  Initialize errors and sdobs
 */
    for (i = 0; i < 4; i++) {
        sp->error[i] = NULLVAL;
        for (j = 0; j < 4; j++) sp->covar[i][j] = NULLVAL;
    }
    sp->sdobs = NULLVAL;
    sp->mindist = sp->maxdist = NULLVAL;
    sp->smajax = sp->sminax = sp->strike = NULLVAL;
/*
 *  Initialize the rest
 */
    sp->ndef = 0;
    sp->nass = 0;
    sp->ndefsta = 0;
    sp->prank = 0;
    sp->depdp = NULLVAL;
    sp->depdp_error = NULLVAL;
    sp->ndp = 0;
    sp->urms = 0.;
    sp->wrms = 0.;
    sp->bodymag = 0.;
    sp->bodymag_uncertainty = 0.;
    sp->nsta_mb = 0;
    sp->nass_mb = 0;
    sp->nmbagency = 0;
    sp->surfmag = 0.;
    sp->surfmag_uncertainty = 0.;
    sp->nsta_ms = 0;
    sp->nass_ms = 0;
    sp->nMsagency = 0;
    return 0;
}

