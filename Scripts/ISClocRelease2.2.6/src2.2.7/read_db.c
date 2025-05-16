#include "iscloc.h"
extern int verbose;
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;
extern int numagencies;
extern char agencies[MAXBUF][AGLEN];
extern struct timeval t0;
extern char indb[24];             /* read data from this DB account, if any */
extern char outdb[24];          /* write results to this DB account, if any */
extern char in_agency[VALLEN];                   /* author for input assocs */
extern char out_agency[VALLEN];    /* author for new hypocentres and assocs */

/*
 * Functions:
 *    get_data
 *    sort_phaserec_db
 */

/*
 * Local functions
 */
static int get_event(EVREC *ep);
static int get_hyp(EVREC *ep, HYPREC h[]);
static int get_pha(EVREC *ep, PHAREC p[]);

#ifdef WITH_DB
extern PGconn *conn;

/*
 *  Title:
 *     get_data
 *  Synopsis:
 *     Reads hypocentre, pahse and amplitude data from database.
 *     Allocates memory for the arrays of structures needed.
 *  Input Arguments:
 *     ep - pointer to event info
 *  Output Arguments:
 *     hp        - array of hypocentre structures
 *     pp[]      - array of phase structures
 *     stamag_mb - array of mb station magnitude structures
 *     stamag_ms - array of MS station magnitude structures
 *     rdmag_mb  - array of mb reading magnitude structures
 *     rdmag_ms  - array of MS reading magnitude structures
 *     mszh      - array of MS reading component magnitude structures
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     main
 *  Calls:
 *     get_event, get_hyp, get_pha
 */
int get_data(EVREC *ep, HYPREC *hp[], PHAREC *pp[],
             STAMAG *stamag_mb[], STAMAG *stamag_ms[],
             RDMAG *rdmag_mb[], RDMAG *rdmag_ms[], MSZH *mszh[])
{
/*
 *  Get prime, numhyps and numphas from hypocenter and event tables
 */
    if (get_event(ep))
        return 1;
    if (verbose)
        fprintf(logfp, "    evid=%d prime=%d numhyps=%d numphas=%d\n",
                ep->evid, ep->prime, ep->numhyps, ep->numphas);
/*
 *  Allocate memory to hypocenter, phase and stamag structures
 */
    *hp = (HYPREC *)calloc(ep->numhyps, sizeof(HYPREC));
    if (*hp == NULL) {
        fprintf(logfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        fprintf(errfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        errorcode = 1;
        return 1;
    }
    *pp = (PHAREC *)calloc(ep->numphas, sizeof(PHAREC));
    if (*pp == NULL) {
        fprintf(logfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        fprintf(errfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        Free(*hp);
        errorcode = 1;
        return 1;
    }
    if ((*stamag_mb = (STAMAG *)calloc(ep->numsta, sizeof(STAMAG))) == NULL) {
        fprintf(logfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        fprintf(errfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        Free(*pp);
        Free(*hp);
        errorcode = 1;
        return 1;
    }
    if ((*stamag_ms = (STAMAG *)calloc(ep->numsta, sizeof(STAMAG))) == NULL) {
        fprintf(logfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        fprintf(errfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        Free(*stamag_mb);
        Free(*pp);
        Free(*hp);
        errorcode = 1;
        return 1;
    }
    if ((*rdmag_mb = (RDMAG *)calloc(ep->numrd, sizeof(RDMAG))) == NULL) {
        fprintf(logfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        fprintf(errfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        Free(*stamag_ms);
        Free(*stamag_mb);
        Free(*pp);
        Free(*hp);
        errorcode = 1;
        return 1;
    }
    if ((*rdmag_ms = (RDMAG *)calloc(ep->numrd, sizeof(RDMAG))) == NULL) {
        fprintf(logfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        fprintf(errfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        Free(*rdmag_mb);
        Free(*stamag_ms);
        Free(*stamag_mb);
        Free(*pp);
        Free(*hp);
        errorcode = 1;
        return 1;
    }
    if ((*mszh = (MSZH *)calloc(ep->numrd, sizeof(MSZH))) == NULL) {
        fprintf(logfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        fprintf(errfp, "get_data: evid %d: cannot allocate memory\n", ep->evid);
        Free(*rdmag_ms);
        Free(*rdmag_mb);
        Free(*stamag_ms);
        Free(*stamag_mb);
        Free(*pp);
        Free(*hp);
        errorcode = 1;
        return 1;
    }
/*
 *  Fill array of hypocenter structures.
 */
    if (get_hyp(ep, *hp)) {
        Free(*mszh);
        Free(*rdmag_ms);
        Free(*rdmag_mb);
        Free(*stamag_ms);
        Free(*stamag_mb);
        Free(*hp);
        Free(*pp);
        return 1;
    }
/*
 *  Fill array of phase structures.
 */
    if (get_pha(ep, *pp)) {
        Free(*mszh);
        Free(*rdmag_ms);
        Free(*rdmag_mb);
        Free(*stamag_ms);
        Free(*stamag_mb);
        Free(*hp);
        Free(*pp);
        return 1;
    }
    return 0;
}

/*
 *  Title:
 *     get_event
 *  Synopsis:
 *     Gets prime, event type, number of hypocenters, phases, readings,
 *     stations and reporting agencies for an event from DB.
 *  Input Arguments:
 *     ep - pointer to structure containing event information.
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     get_data
 *  Calls:
 *     pgsql_error, dropspace
 */
static int get_event(EVREC *ep)
{
    int i;
    PGresult *res_set = (PGresult *)NULL;
    char psql[2048], sql[2048];
    ep->isc_hypid = NULLVAL;
    ep->outdbprime = 0;
    ep->outdbisc_hypid = 0;
/*
 *  Get hypid of ISC hypocentre if it exists
 */
    sprintf(psql, "SELECT hypid              \
                     FROM %shypocenter       \
                    WHERE deprecated is NULL \
                      AND author = '%s'      \
                      AND isc_evid = %d", indb, in_agency, ep->evid);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            get_event: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("get_event: error: isc hypocentre");
    else {
        if (PQntuples(res_set) == 0)
            ep->isc_hypid = NULLVAL;
        else if (PQntuples(res_set) == 1)
            ep->isc_hypid = atoi(PQgetvalue(res_set, 0, 0));
        else {
/*
 *          if more than one ISC hypocentre exists,
 *          find the one with association records
 */
            PQclear(res_set);
            sprintf(psql, "SELECT distinct h.hypid     \
                             FROM %shypocenter h, %sassociation a \
                            WHERE h.deprecated is NULL \
                              AND h.author = '%s'      \
                              AND h.author = a.author  \
                              AND h.hypid = a.hypid    \
                              AND h.isc_evid = %d",
                    indb, indb, in_agency, ep->evid);
            dropspace(psql, sql);
            if (verbose > 2) fprintf(logfp, "            get_event: %s\n", sql);
            if ((res_set = PQexec(conn, sql)) == NULL)
                pgsql_error("get_event: error: isc hypocentre");
            else
                ep->isc_hypid = atoi(PQgetvalue(res_set, 0, 0));
        }
    }
    PQclear(res_set);
    if (verbose > 1) fprintf(logfp, "        isc_hypid=%d\n", ep->isc_hypid);
/*
 *  Get hypid and etype of prime hypocentre
 */
    sprintf(psql, "SELECT prime_hyp, COALESCE(etype, 'ke') \
                     FROM %sevent WHERE evid = %d", indb, ep->evid);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            get_event: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("get_event: error: prime hypocentre");
    else if (PQntuples(res_set) == 1) {
        ep->prime = atoi(PQgetvalue(res_set, 0, 0));
        strcpy(ep->etype, PQgetvalue(res_set, 0, 1));
        PQclear(res_set);
    }
    else {
        PQclear(res_set);
        fprintf(errfp, "get_event: %d no prime found\n", ep->evid);
        fprintf(logfp, "get_event: %d no prime found\n", ep->evid);
        return 1;
    }
/*
 *  Get prime hypid and isc_hypid in outdb if it differs from indb
 */
    if (strcmp(indb, outdb)) {
        ep->outdbprime = 0;
        sprintf(psql, "SELECT prime_hyp FROM %sevent WHERE evid = %d",
                outdb, ep->evid);
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            get_event: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL) {
            pgsql_error("get_event: error: prime hypocentre");
        }
        else if (PQntuples(res_set) == 1) {
            ep->outdbprime = atoi(PQgetvalue(res_set, 0, 0));
        }
        else {
            ep->outdbprime = 0;
        }
        PQclear(res_set);
        sprintf(psql, "SELECT hypid FROM %shypocenter       \
                        WHERE author = '%s' AND isc_evid = %d",
                outdb, out_agency, ep->evid);
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            get_event: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL) {
            pgsql_error("get_event: error: isc hypocentre");
        }
        else if (PQntuples(res_set) == 1) {
            ep->outdbisc_hypid = atoi(PQgetvalue(res_set, 0, 0));
        }
        else {
            ep->outdbisc_hypid = 0;
        }
        PQclear(res_set);
    }
/*
 *  Get hypocentre count
 */
    sprintf(psql, "SELECT COUNT(hypid)                             \
                     FROM %shypocenter                             \
                    WHERE (deprecated is NULL OR deprecated = 'M') \
                      AND hypid = pref_hypid                       \
                      AND isc_evid = %d", indb, ep->evid);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            get_event: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("get_event: error: hypocentre count");
    else {
        ep->numhyps = atoi(PQgetvalue(res_set, 0, 0));
    }
    PQclear(res_set);
    if (ep->numhyps == 0) {
        fprintf(errfp, "get_event %d: no hypocentres found\n", ep->evid);
        fprintf(logfp, "get_event %d: no hypocentres found\n", ep->evid);
        return 1;
    }
    if (verbose > 1) fprintf(logfp, "        %d hypocentres\n", ep->numhyps);
/*
 *  Get phase, reading and station count
 */
    sprintf(psql, "SELECT COUNT(p.phid), COUNT(distinct p.rdid),       \
                          COUNT(distinct a.sta)                        \
                     FROM %sphase p, %sassociation a                   \
                    WHERE a.hypid = %d                                 \
                      AND a.author = '%s'                              \
                      AND p.phid = a.phid                              \
                      AND (p.deprecated is NULL OR p.deprecated = 'M') \
                      AND  a.deprecated is NULL",
            indb, indb, ep->prime, in_agency);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            get_event: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("get_event: error: phase count");
    else {
        ep->numphas = atoi(PQgetvalue(res_set, 0, 0));
        ep->numrd = atoi(PQgetvalue(res_set, 0, 1));
        ep->numsta = atoi(PQgetvalue(res_set, 0, 2));
    }
    PQclear(res_set);
    if (ep->numphas == 0) {
        fprintf(errfp, "get_event %d: no phases found\n", ep->evid);
        fprintf(logfp, "get_event %d: no phases found\n", ep->evid);
        return 1;
    }
    if (ep->numrd == 0) {
        fprintf(errfp, "get_event %d: no readings found\n", ep->evid);
        fprintf(logfp, "get_event %d: no readings found\n", ep->evid);
        return 1;
    }
    if (ep->numsta == 0) {
        fprintf(errfp, "get_event %d: no stations found\n", ep->evid);
        fprintf(logfp, "get_event %d: no stations found\n", ep->evid);
        return 1;
    }
    if (verbose > 1)
        fprintf(logfp, "        %d phases, %d readings, %d stations\n",
                ep->numphas, ep->numrd, ep->numsta);
/*
 *  Get agencies that reported phases
 */
    sprintf(psql, "SELECT distinct r.reporter                          \
                     FROM %sphase p, %sassociation a, %sreport r       \
                    WHERE a.hypid = %d                                 \
                      AND a.author = '%s'                              \
                      AND p.phid = a.phid                              \
                      AND r.repid = p.reporter                         \
                      AND (p.deprecated is NULL OR p.deprecated = 'M') \
                      AND  a.deprecated is NULL",
            indb, indb, indb, ep->prime, in_agency);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            get_event: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("get_event: error: agency count");
    else {
        ep->numagency = numagencies = PQntuples(res_set);
        if (ep->numagency == 0) {
            fprintf(errfp, "get_event %d: no agencies found\n", ep->evid);
            fprintf(logfp, "get_event %d: no agencies found\n", ep->evid);
            PQclear(res_set);
            return 1;
        }
        for (i = 0; i < numagencies; i++)
            strcpy(agencies[i], PQgetvalue(res_set, i, 0));
        PQclear(res_set);
    }
    if (verbose > 1) fprintf(logfp, "        %d agencies\n", ep->numagency);
    return 0;
}

/*
 *  Title:
 *     get_hyp
 *  Synopsis:
 *     Reads reported hypocentres for an event from DB.
 *         Fields from the hypocenter table:
 *            hypid
 *            origin epoch time
 *            lat, lon
 *            depth (depdp if depth is null)
 *            depfix, epifix, timfix
 *            nsta, ndefsta, nass, ndef
 *            mindist, maxdist, azimgap
 *            author
 *        Fields from the hypoc_err table:
 *            hypid
 *            sminax, smajax, strike
 *            stime, sdepth
 *            sdobs
 *        Fields from the hypoc_acc table:
 *            hypid
 *            score
 *     Sorts hypocentres by score, the first being the prime hypocentre.
 *  Input Arguments:
 *     ep  - pointer to event info
 *     h[] - array of hypocentres.
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     get_data
 * Calls:
 *     pgsql_error, dropspace
 */
static int get_hyp(EVREC *ep, HYPREC h[])
{
    PGresult *res_set = (PGresult *)NULL;
    HYPREC temp;
    char sql[2048], psql[2048];
    int numhyps = 0, numerrs = 0;
    int i, j, row;
    int hypid = 0, score = 0;
    int sec = 0, msec = 0;
    double minax = 0., majax = 0., strike = 0., stime = 0.;
    double sdepth = 0., sdobs = 0.;
/*
 *  Get hypocenters
 */
    sprintf(psql, "SELECT hypid, EXTRACT(EPOCH FROM day), COALESCE(msec, 0), \
                          lat, lon, COALESCE(depth, depdp),                  \
                          COALESCE(depfix, '0'), COALESCE(epifix, '0'),      \
                          COALESCE(timfix, '0'), nsta, ndefsta, nass, ndef,  \
                          mindist, maxdist, azimgap, author                  \
                     FROM %shypocenter                                       \
                    WHERE (deprecated is NULL OR deprecated = 'M')           \
                      AND hypid = pref_hypid                                 \
                      AND isc_evid = %d                                      \
                 ORDER BY hypid", indb, ep->evid);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            get_hyp: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("get_hyp: hypocentre");
    else {
        numhyps = PQntuples(res_set);
        for (i = 0; i < numhyps; i++) {
            h[i].depfix = 0;
            h[i].epifix = 0;
            h[i].timfix = 0;
            h[i].rank = 0;
            strcpy(h[i].etype, ep->etype);
            h[i].sminax = NULLVAL;
            h[i].smajax = NULLVAL;
            h[i].strike = NULLVAL;
            h[i].stime = NULLVAL;
            h[i].sdepth = NULLVAL;
            h[i].sdobs = NULLVAL;
            h[i].hypid = atoi(PQgetvalue(res_set, i, 0));
            sec  = atoi(PQgetvalue(res_set, i, 1));
            msec = atoi(PQgetvalue(res_set, i, 2));
            h[i].time = (double)sec + (double)msec / 1000.;
            h[i].lat  = atof(PQgetvalue(res_set, i, 3));
            h[i].lon  = atof(PQgetvalue(res_set, i, 4));
            if (PQgetisnull(res_set, i, 5))       { h[i].depth = NULLVAL; }
            else          { h[i].depth = atof(PQgetvalue(res_set, i, 5)); }
            if (strcmp(PQgetvalue(res_set, i, 6), "0"))  h[i].depfix = 1;
            if (strcmp(PQgetvalue(res_set, i, 7), "0"))  h[i].epifix = 1;
            if (strcmp(PQgetvalue(res_set, i, 8), "0"))  h[i].timfix = 1;
            if (PQgetisnull(res_set, i, 9))        { h[i].nsta = NULLVAL; }
            else           { h[i].nsta = atoi(PQgetvalue(res_set, i, 9)); }
            if (PQgetisnull(res_set, i, 10))    { h[i].ndefsta = NULLVAL; }
            else       { h[i].ndefsta = atoi(PQgetvalue(res_set, i, 10)); }
            if (PQgetisnull(res_set, i, 11))       { h[i].nass = NULLVAL; }
            else          { h[i].nass = atoi(PQgetvalue(res_set, i, 11)); }
            if (PQgetisnull(res_set, i, 12))       { h[i].ndef = NULLVAL; }
            else          { h[i].ndef = atoi(PQgetvalue(res_set, i, 12)); }
            if (PQgetisnull(res_set, i, 13))    { h[i].mindist = NULLVAL; }
            else       { h[i].mindist = atof(PQgetvalue(res_set, i, 13)); }
            if (PQgetisnull(res_set, i, 14))    { h[i].maxdist = NULLVAL; }
            else       { h[i].maxdist = atof(PQgetvalue(res_set, i, 14)); }
            if (PQgetisnull(res_set, i, 15))    { h[i].azimgap = NULLVAL; }
            else       { h[i].azimgap = atof(PQgetvalue(res_set, i, 15)); }
            strcpy(h[i].agency, PQgetvalue(res_set, i, 16));
            if (verbose > 1)
                fprintf(logfp, "        i=%d evid=%d hypid=%d agency=%s\n",
                        i, ep->evid, h[i].hypid, h[i].agency);
        }
    }
    PQclear(res_set);
    if (numhyps != ep->numhyps) {
        fprintf(errfp, "get_hyp: unexpected numhyps %d\n", numhyps);
        fprintf(logfp, "get_hyp: unexpected numhyps %d\n", numhyps);
        return 1;
    }
/*
 *  Get hypocenter uncertainties
 */
    sprintf(psql, "SELECT e.hypid, sminax, smajax, strike, \
                          stime, sdepth, sdobs             \
                     FROM %shypocenter h, %shypoc_err e    \
                    WHERE isc_evid = %d                    \
                      AND e.hypid = h.hypid                \
                      AND h.hypid = h.pref_hypid           \
                      AND (h.deprecated is NULL OR h.deprecated = 'M') \
                 ORDER BY h.hypid;", indb, indb, ep->evid);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            get_hyp: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("get_hyp: hypoc_err");
    else {
        numerrs = PQntuples(res_set);
        for (row = 0; row < numerrs; row++) {
            hypid = atoi(PQgetvalue(res_set, row, 0));
            if (PQgetisnull(res_set, row, 1))      { minax = NULLVAL; }
            else         { minax = atof(PQgetvalue(res_set, row, 1)); }
            if (PQgetisnull(res_set, row, 2))      { majax = NULLVAL; }
            else         { majax = atof(PQgetvalue(res_set, row, 2)); }
            if (PQgetisnull(res_set, row, 3))     { strike = NULLVAL; }
            else        { strike = atof(PQgetvalue(res_set, row, 3)); }
            if (PQgetisnull(res_set, row, 4))      { stime = NULLVAL; }
            else         { stime = atof(PQgetvalue(res_set, row, 4)); }
            if (PQgetisnull(res_set, row, 5))     { sdepth = NULLVAL; }
            else        { sdepth = atof(PQgetvalue(res_set, row, 5)); }
            if (PQgetisnull(res_set, row, 6))      { sdobs = NULLVAL; }
            else         { sdobs = atof(PQgetvalue(res_set, row, 6)); }
            for (i = 0; i < numhyps; i++) {
                if (hypid == h[i].hypid) {
                    h[i].sminax = minax;
                    h[i].smajax = majax;
                    h[i].strike = strike;
                    h[i].stime = stime;
                    h[i].sdepth = sdepth;
                    h[i].sdobs = sdobs;
                    break;
                }
            }
        }
    }
    PQclear(res_set);
/*
 *  Get hypocenter scores
 */
    sprintf(psql, "SELECT h.hypid, a.score              \
                     FROM %shypocenter h, %shypoc_acc a \
                    WHERE h.isc_evid = %d               \
                      AND a.hypid = h.hypid             \
                      AND h.hypid = h.pref_hypid        \
                      AND (h.deprecated is NULL OR h.deprecated = 'M') \
                 ORDER BY h.hypid;", indb, indb, ep->evid);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            get_hyp: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("get_hyp: hypoc_acc");
    else {
        numerrs = PQntuples(res_set);
        for (row = 0; row < numerrs; row++) {
            hypid = atoi(PQgetvalue(res_set, row, 0));
            if (PQgetisnull(res_set, row, 1))       { score = 0.; }
            else     { score = atof(PQgetvalue(res_set, row, 1)); }
            for (i = 0; i < numhyps; i++) {
                if (hypid == h[i].hypid) {
                    h[i].rank = score;
                    break;
                }
            }
        }
    }
    PQclear(res_set);
/*
 *  set h[0] to prime
 */
    for (i = 1; i < numhyps; i++) {
        if (h[i].hypid == ep->prime) {
            swap(h[i], h[0]);
            break;
        }
    }
/*
 *  sort the rest by score
 */
    for (i = 1; i < numhyps - 1; i++) {
        for (j = i + 1; j < numhyps; j++) {
            if (h[i].rank < h[j].rank) {
                swap(h[i], h[j]);
            }
        }
    }
    return 0;
}

/*
 *  Title:
 *     get_pha
 *  Synopsis:
 *     Reads phase, amplitude, and reporter association data for an event
 *         from DB.
 *     Performs three seperate queries as phases may or may not be associated
 *         by their reporter and may or may not have one or more amplitudes.
 *     Reads phases ISC associated to the prime hypocentre
 *         Fields from the association, phase, site and reporter tables:
 *             phase.phid
 *             phase.rdid
 *             phase.day, phase.msec (arrival epoch time)
 *             phase.phase (reported phase)
 *             phase.sta
 *             phase.slow, phase.azim
 *             phase.chan, phase.sp_fm, phase.emergent, phase.impulsive
 *             association.reporter (reporter id)
 *             association.phase_fixed
 *             site.lat, site.lon, site.elev
 *             site.prista (site.sta if prista is null)
 *             reporter.reporter (reporting agency)
 *             association.delta
 *             association.phase (ISC phase)
 *             association.nondef
 *     Reads reported associations for any of the phases selected above.
 *         Replaces reported phase name with the reported association if it was
 *         reidentified by NEIC, CSEM, EHB or IASPEI.
 *     Reads amplitudes associated to the prime hypocentre.
 *         Fields from the amplitude table:
 *             phid
 *             amp, per
 *             logat
 *             amptype
 *             chan
 *             ampid
 *     Sorts phase structures so that they ordered by delta, prista, rdid, time.
 *         Uses prista rather than sta so that duplicates weighted properly.
 *     Marks initial arrivals in readings.
 *  Input Arguments:
 *     ep  - pointer to event info
 *     p[] - array of phase structures
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     get_data
 *  Calls:
 *     pgsql_error, dropspace
 */
static int get_pha(EVREC *ep, PHAREC p[])
{
    char sql[2048], psql[2048];
    int i, m, row, prev_rdid;
    int numphas = 0, numamps = 0, numrepas = 0, rephypset = 0;
    int ass_phid = 0, amp_phid = 0, amp_ampid = 0, hypid = 0;
    int sec = 0, msec = 0, emergent = 0, impulsive = 0;
    double amp = 0., per = 0., logat = 0.;
    char ass_phase[PHALEN], chan[10], w[10];
    char amptype[10], ampchan[10];
    PGresult *res_set = (PGresult *)NULL;
/*
 *  Get phases ISC associated to the prime hyp
 *     site.net is NULL for registered stations!
 */
    sprintf(psql, "SELECT p.phid, rdid,                  \
                          EXTRACT(EPOCH FROM day), msec, \
                          COALESCE(p.phase, ''),         \
                          p.sta, p.slow, p.azim,         \
                          COALESCE(chan, '   '),         \
                          COALESCE(sp_fm, ' '),          \
                          COALESCE(emergent, '0'),       \
                          COALESCE(impulsive, '0'),      \
                          a.reporter,                    \
                          COALESCE(phase_fixed, '0'),    \
                          s.lat, s.lon, s.elev,          \
                          COALESCE(prista, s.sta),       \
                          COALESCE(r.reporter, ''),      \
                          delta,                         \
                          COALESCE(a.phase, ''),         \
                          COALESCE(a.nondef, '0')        \
                     FROM %sphase p, %ssite s,           \
                          %sassociation a, %sreport r    \
                    WHERE hypid = %d                     \
                      AND a.author = '%s'                \
                      AND a.deprecated is NULL           \
                      AND a.phid = p.phid                \
                      AND (p.deprecated is NULL OR p.deprecated = 'M') \
                      AND s.sta = p.sta                  \
                      AND s.net is NULL                  \
                      AND p.reporter = r.repid           \
                 ORDER BY p.phid",
            indb, indb, indb, indb, ep->prime, in_agency);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            get_pha: %s\n", sql);
    if ((res_set = PQexec(conn, psql)) == NULL)
        pgsql_error("get_pha: phases");
    else {
        numphas = PQntuples(res_set);
        /* Check for something very odd - */
        /* get_event does same query to fill ep->numphas. */
        if (numphas != ep->numphas) {
            fprintf(errfp, "get_pha: unexpected number phases!\n");
            fprintf(errfp, "    evid=%d prime=%d epnumphas=%d numphas=%d\n",
                    ep->evid, ep->prime, ep->numphas, numphas);
            fprintf(errfp, "    often due to a missing station in site\n");
            fprintf(logfp, "get_pha: unexpected number phases!\n");
            fprintf(logfp, "    evid=%d prime=%d epnumphas=%d numphas=%d\n",
                    ep->evid, ep->prime, ep->numphas, numphas);
            fprintf(logfp, "    often due to a missing station in site\n");
            PQclear(res_set);
            return 1;
        }
        for (i = 0; i < numphas; i++) {
            strcpy(p[i].arrid, PQgetvalue(res_set, i, 0));
            p[i].phid = atoi(PQgetvalue(res_set, i, 0));
            p[i].rdid = atoi(PQgetvalue(res_set, i, 1));
            if (verbose > 3)
                fprintf(logfp, "        i=%d phid=%d rdid=%d\n",
                        i, p[i].phid, p[i].rdid);
            if (PQgetisnull(res_set, i, 2))         { sec = NULLVAL; }
            else            { sec = atoi(PQgetvalue(res_set, i, 2)); }
            if (PQgetisnull(res_set, i, 3))              { msec = 0; }
            else           { msec = atoi(PQgetvalue(res_set, i, 3)); }
            /* arrival epoch time */
            if (sec != NULLVAL)
                p[i].time = (double)sec + (double)msec / 1000.;
            else
                p[i].time = NULLVAL;
            strcpy(p[i].rep_phase, PQgetvalue(res_set, i, 4));
            strcpy(p[i].sta, PQgetvalue(res_set, i, 5));
            if (PQgetisnull(res_set, i, 6))        { p[i].slow = NULLVAL; }
            else           { p[i].slow = atof(PQgetvalue(res_set, i, 6)); }
            if (PQgetisnull(res_set, i, 7))        { p[i].azim = NULLVAL; }
            else           { p[i].azim = atof(PQgetvalue(res_set, i, 7)); }
            strcpy(chan,  PQgetvalue(res_set, i, 8));
            strcpy(p[i].pch, chan);
            if (chan[2] != '?') p[i].comp = chan[2];
            else                p[i].comp = ' ';
            strcpy(w, PQgetvalue(res_set, i, 9));
            p[i].sp_fm = w[0];
            strcpy(w, PQgetvalue(res_set, i, 10));
            emergent = (streq(w, "0")) ? 0 : 1;
            strcpy(w, PQgetvalue(res_set, i, 11));
            impulsive = (streq(w, "0")) ? 0 : 1;
            if (emergent && impulsive) p[i].detchar = 'q';
            else if (impulsive)        p[i].detchar = 'i';
            else if (emergent)         p[i].detchar = 'e';
            else                       p[i].detchar = ' ';
            p[i].repid = atoi(PQgetvalue(res_set, i, 12));
            strcpy(w, PQgetvalue(res_set, i, 13));
            p[i].phase_fixed = (streq(w, "0")) ? 0 : 1;
            p[i].sta_lat = atof(PQgetvalue(res_set, i, 14));
            p[i].sta_lon = atof(PQgetvalue(res_set, i, 15));
            if (PQgetisnull(res_set, i, 16))        { p[i].sta_elev = 0.; }
            else      { p[i].sta_elev = atof(PQgetvalue(res_set, i, 16)); }
            strcpy(p[i].prista, PQgetvalue(res_set, i, 17));
            strcpy(p[i].rep, PQgetvalue(res_set, i, 18));
            strcpy(p[i].auth, p[i].rep);
            strcpy(p[i].agency, p[i].rep);
            if (PQgetisnull(res_set, i, 19))      { p[i].delta = NULLVAL; }
            else         { p[i].delta = atof(PQgetvalue(res_set, i, 19)); }
            strcpy(p[i].phase, PQgetvalue(res_set, i, 20));
            strcpy(w, PQgetvalue(res_set, i, 21));
            p[i].force_undef = (streq(w, "0")) ? 0 : 1;
            p[i].numamps = 0;
            p[i].purged = 0;           /* used by iscloc_search */
//            strcpy(p[i].agency, "FDSN");
            strcpy(p[i].deploy, "IR");
            strcpy(p[i].lcn, "--");
            sprintf(p[i].fdsn, "FDSN.IR.%s.--", p[i].sta);
        }
    }
    PQclear(res_set);
/*
 *  Select reported associations for any of the phases selected above.
 *  Use phase from reported association over one from phase in case
 *      re-identified by NEIC, CSEM, EHB or IASPEI
 */
    sprintf(psql, "SELECT a1.hypid, COALESCE(a1.phase, ''), a1.phid     \
                     FROM %sassociation a1, %sassociation a2, %sphase p \
                    WHERE a2.hypid = %d                                 \
                      AND a2.author = '%s'                              \
                      AND a2.deprecated is NULL                         \
                      AND p.phid = a2.phid                              \
                      AND (p.deprecated is NULL OR p.deprecated = 'M')  \
                      AND a1.phid = a2.phid                             \
                      AND a1.author != '%s'                             \
                 ORDER BY a1.phid",
            indb, indb, indb, ep->prime, in_agency, in_agency);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            get_pha: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("get_pha: reported associations");
    else {
        numrepas = PQntuples(res_set);
        for (row = 0; row < numrepas; row++) {
            hypid = atoi(PQgetvalue(res_set, row, 0));
            strcpy(ass_phase, PQgetvalue(res_set, row, 1));
            ass_phid = atoi(PQgetvalue(res_set, row, 2));
            for (i = 0; i < numphas; i++) {
                rephypset = 1;
                if (ass_phid == p[i].phid &&
                    (streq(p[i].agency, "NEIC") ||
                     streq(p[i].agency, "NEIS") ||
                     streq(p[i].agency, "EHB")  ||
                     streq(p[i].agency, "CSEM") ||
                     streq(p[i].agency, "IASPEI"))) {
                    p[i].hypid = hypid;
                    rephypset = 0;
                    if (strlen(ass_phase) > 0)
                        strcpy(p[i].rep_phase, ass_phase);
                }
                if (rephypset)
                    p[i].hypid = NULLVAL;
            }
        }
    }
    PQclear(res_set);
/*
 *  get amplitudes
 */
    sprintf(psql, "SELECT am.phid, am.amp, am.per, am.logat,   \
                          COALESCE(am.amptype, ''),            \
                          COALESCE(am.chan, '   '), am.ampid   \
                     FROM %samplitude am, %sassociation a, %sphase p \
                    WHERE a.hypid = %d                         \
                      AND a.author = '%s'                      \
                      AND a.deprecated is NULL                 \
                      AND p.phid = a.phid                      \
                      AND (p.deprecated is NULL OR p.deprecated = 'M') \
                      AND am.phid = p.phid                     \
                      AND am.deprecated is NULL                \
                 ORDER BY am.phid",
            indb, indb, indb, ep->prime, in_agency);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            get_pha: %s\n", sql);
    if ((res_set = PQexec(conn, psql)) == NULL)
        pgsql_error("get_pha: amplitudes");
    else {
        numamps = PQntuples(res_set);
        for (row = 0; row < numamps; row++) {
            amp_phid = atoi(PQgetvalue(res_set, row, 0));
            if (PQgetisnull(res_set, row, 1))        { amp = NULLVAL; }
            else           { amp = atof(PQgetvalue(res_set, row, 1)); }
            if (PQgetisnull(res_set, row, 2))        { per = NULLVAL; }
            else           { per = atof(PQgetvalue(res_set, row, 2)); }
            if (PQgetisnull(res_set, row, 3))      { logat = NULLVAL; }
            else         { logat = atof(PQgetvalue(res_set, row, 3)); }
            strcpy(amptype, PQgetvalue(res_set, row, 4));
            strcpy(ampchan, PQgetvalue(res_set, row, 5));
            amp_ampid = atoi(PQgetvalue(res_set, row, 6));
            for (i = 0; i < numphas; i++) {
                m = p[i].numamps;
                if (amp_phid == p[i].phid) {
/*
 *                  by default, amplitudes are interpreted as zero-to-peak
 *                  0-to-p and p-to-p are interpreted,
 *                  everything else is ignored
 */
                    if (streq("0-to-p", amptype))
                        p[i].a[m].amp = amp;
                    else if (streq("p-to-p", amptype))
                        p[i].a[m].amp = amp / 2.;
                    else if (streq("velocity", amptype))
                        continue;
                    else if (streq("s", amptype))
                        continue;
                    else if (strlen(amptype))
                        continue;
                    else
                        p[i].a[m].amp = amp;
                    p[i].a[m].ampid = amp_ampid;
                    p[i].a[m].per = per;
                    p[i].a[m].logat = logat;
/*
 *                  If no component given for amplitude then use one from phase
 */
                    strcpy(p[i].a[m].ach, ampchan);
                    if (ampchan[2] == 'Z' ||
                        ampchan[2] == 'E' ||
                        ampchan[2] == 'N')  p[i].a[m].comp = ampchan[2];
                    else if (p[i].comp)     p[i].a[m].comp = p[i].comp;
                    else                    p[i].a[m].comp = ' ';
/*
 *                  initializations
 */
                    p[i].a[m].ampdef = p[i].a[m].magid = 0;
                    p[i].a[m].magnitude = NULLVAL;
                    strcpy(p[i].a[m].magtype, "");
                    p[i].numamps++;
                }
            }
        }
    }
    PQclear(res_set);
/*
 *  Sort phase structures so that they ordered by delta, prista, rdid, time.
 *  prista rather than sta used so that duplicates weighted properly.
 */
    if (verbose > 2)
        fprintf(logfp, "            get_pha: sort phases (%.2f)\n", secs(&t0));
    sort_phaserec_db(numphas, p);
/*
 *  mark initial arrivals in readings
 */
    prev_rdid = -1;
    for (i = 0; i < numphas; i++) {
        p[i].init = 0;
        if (p[i].rdid != prev_rdid)
            p[i].init = 1;
        prev_rdid = p[i].rdid;
    }
    if (verbose > 2)
        print_pha(numphas, p);
    return 0;
}

#endif  /* WITH_DB */

/*
 *  Title:
 *     sort_phaserec_db
 *  Synopsis:
 *     Sorts phase structures so that they ordered by delta, prista, rdid, time.
 *  Input Arguments:
 *     numphas    - number of associated phases
 *     p[]        - array of phase structures.
 *  Called by:
 *     get_pha, locate_event
 */
void sort_phaserec_db(int numphas, PHAREC p[])
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
            if (p[j].rdid > p[j+1].rdid) {
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

/*  EOF */
