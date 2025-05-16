#include "iscloc.h"
extern int verbose;                                        /* verbose level */
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;
extern struct timeval t0;
extern char indb[24];             /* read data from this DB account, if any */
extern char outdb[24];          /* write results to this DB account, if any */
extern char out_agency[VALLEN];    /* author for new hypocentres and assocs */
extern char in_agency[VALLEN];                   /* author for input assocs */
extern int repid;                                            /* reporter id */

/*
 * Functions:
 *    put_data
 *    put_hypoc_err
 *    put_hypoc_acc
 *    put_netmag
 *    put_stamag
 *    put_rdmag
 *    put_assoc
 *    put_ampmag
 *    put_mszh
 *    remove_isc
 *    replace_prime
 *    replace_assoc
 *    get_id
 */

/*
 * Local functions
 */
static void update_event(EVREC *ep);
static void put_event(EVREC *ep);
static int put_hypocenter(EVREC *ep, SOLREC *sp, FE *fep);
static void delete_isc_hypid(int isc_hypid);
static void delete_assoc(int old_prime, char *author);

#ifdef WITH_DB
extern PGconn *conn;

/*
 *  Title:
 *     put_data
 *  Synopsis:
 *     Writes solution to database.
 *     Removes old ISC solution if any, and populates hypocenter table.
 *     Populates or updates event table if necessary.
 *     Populates hypoc_err, hypoc_acc and network_quality tables.
 *     Deletes previous ISC associations and populates association table.
 *     Populates netmag, stamag, rdmag, ampmag and ms_zh tables.
 *  Input Arguments:
 *     ep        - pointer to event info
 *     sp        - pointer to current solution
 *     p[]       - array of phase structures
 *     hq        - pointer to hypocentre quality structure
 *     stamag_mb - array of mb stamag structures
 *     stamag_ms - array of Ms stamag structures
 *     fep       - pointer to Flinn_Engdahl region number structure
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     eventloc
 *  Calls:
 *     put_hyp, put_event, put_pha
 */
int put_data(EVREC *ep, SOLREC *sp, PHAREC p[], HYPQUAL *hq,
             STAMAG *stamag_mb, STAMAG *stamag_ms,
             RDMAG *rdmag_mb, RDMAG *rdmag_ms, MSZH *mszh, FE *fep)
{
    int hypid = ep->prime, old_isc_hypid = ep->isc_hypid;
    char author[VALLEN];
    strcpy(author, in_agency);
    if (verbose)
        fprintf(logfp, "        put_data: prev_hypid=%d evid=%d\n",
                ep->isc_hypid, ep->evid);
/*
 *  populate hypocenter table; remove old ISC solution
 */
    if (verbose) fprintf(logfp, "        put_hypocenter (%.2f)\n", secs(&t0));
    if (put_hypocenter(ep, sp, fep))
        return 1;
    if (strcmp(indb, outdb) && !ep->outdbprime) {
/*
 *      populate event table in outdb if necessary
 */
        if (verbose) fprintf(logfp, "        put_event (%.2f)\n", secs(&t0));
        put_event(ep);
    }
    else if (old_isc_hypid == NULLVAL) {
/*
 *      update event table if new hypocentre
 */
        if (verbose) fprintf(logfp, "        update_event (%.2f)\n", secs(&t0));
        update_event(ep);
    }
    sp->hypid = ep->isc_hypid;
/*
 *  populate hypoc_err table
 */
    if (verbose) fprintf(logfp, "        put_hypoc_err (%.2f)\n", secs(&t0));
    put_hypoc_err(ep, sp);
/*
 *  populate hypoc_acc and network_quality tables
 */
    if (verbose) fprintf(logfp, "        put_hypoc_acc (%.2f)\n", secs(&t0));
    put_hypoc_acc(ep, sp, hq);
/*
 *  delete previous associations
 */
    if (verbose) fprintf(logfp, "        delete_assoc (%.2f)\n", secs(&t0));
    if (strcmp(indb, outdb) && ep->outdbprime) {
        hypid = ep->outdbprime;
        strcpy(author, out_agency);
    }
    delete_assoc(hypid, author);
/*
 *  populate association table
 */
    if (verbose) fprintf(logfp, "        put_assoc (%.2f)\n", secs(&t0));
    put_assoc(sp, p);
/*
 *  populate netmag table
 */
    sp->ms_id = sp->mb_id = NULLVAL;
    if (sp->surfmag != NULLVAL || sp->bodymag != NULLVAL) {
        if (verbose) fprintf(logfp, "        put_netmag (%.2f)\n", secs(&t0));
        if (put_netmag(sp))
            return 1;
    }
/*
 *  populate stamag table
 */
    if (sp->nsta_mb || sp->nsta_ms) {
        if (verbose) fprintf(logfp, "        put_stamag (%.2f)\n", secs(&t0));
        if (put_stamag(sp, stamag_mb, stamag_ms))
            return 1;
    }
/*
 *  populate rdmag and ampmag tables
 */
    if (sp->nass_mb || sp->nass_ms) {
        if (verbose) fprintf(logfp, "        put_rdmag (%.2f)\n", secs(&t0));
        if (put_rdmag(sp, rdmag_mb, rdmag_ms))
            return 1;
        if (verbose) fprintf(logfp, "        put_ampmag (%.2f)\n", secs(&t0));
        if (put_ampmag(sp, p))
            return 1;
    }
/*
 *  populate mszh table
 */
    if (sp->nass_ms) {
        if (verbose) fprintf(logfp, "        put_mszh (%.2f)\n", secs(&t0));
        if (put_mszh(sp, mszh))
            return 1;
    }
/*
 *  set prime hypid
 */
    ep->prime = ep->isc_hypid;
    if (verbose)
        fprintf(logfp, "        put_data: done (%.2f)\n", secs(&t0));
    return 0;
}

/*
 *  Title:
 *     update_event
 *  Synopsis:
 *     Updates prime_hyp in the event table.
 *     Only called if the ISC hypocentre is new.
 *  Input Arguments:
 *     ep - pointer to event info
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     put_data
 *  Calls:
 *     pgsql_error, dropspace
 */
static void update_event(EVREC *ep)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[1024], psql[1024], errmsg[1024];
/*
 *  Update prime_hyp
 */
    sprintf(psql, "UPDATE %sevent SET prime_hyp = %d, moddate = NOW() \
                    WHERE evid = %d", outdb, ep->isc_hypid, ep->evid);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            update_event: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("update_event:");
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "update_event: %d", PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
}

/*
 *  Title:
 *     put_event
 *  Synopsis:
 *     Populates event table in outdb if event is not yet there
 *  Input Arguments:
 *     ep - pointer to event info
 *  Called by:
 *     put_data
 *  Calls:
 *     pgsql_error, dropspace
 */
static void put_event(EVREC *ep)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[1024], psql[1024], errmsg[1024];
    sprintf(psql, "INSERT INTO %sevent                  \
                   (author, evid, lddate, moddate,      \
                    prime_hyp, ready, reporter, etype)  \
                   VALUES ('%s', %d, NOW(), NOW(), %d, 'R', %d, '%s')",
            outdb, out_agency, ep->evid, ep->isc_hypid, repid, ep->etype);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            put_event: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("put_event:");
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "put_event: %d", PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
}

/*
 *  Title:
 *     put_hypocenter
 *  Synopsis:
 *     Populates hypocenter table.
 *     Gets a new hypid if previous prime was not ISC.
 *     Deletes previous prime if it was ISC.
 *     Sets depfix flag:
 *         null: Free-depth solution
 *            A: Depth fixed by ISC Analyst
 *            S: Anthropogenic event; depth fixed to surface
 *            G: Depth fixed to ISC default depth grid
 *            R: Depth fixed to ISC default region depth
 *            M: Depth fixed to median depth of reported hypocentres
 *            B: Beyond depth limits; depth fixed to 0/600 km
 *            H: Depth fixed to depth of a reported hypocentre
 *            D: Depth fixed to depth-phase depth
 *  Input Arguments:
 *     ep  - pointer to event info
 *     sp  - pointer to current solution
 *     fep - pointer to Flinn_Engdahl region number structure
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     put_data
 *  Calls:
 *     pgsql_error, dropspace, get_id, gregnum, gregtosreg
 */
static int put_hypocenter(EVREC *ep, SOLREC *sp, FE *fep)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[2048], psql[2048], errmsg[2048];
    char s[100], depfix[2];
    int nextid = 0, day = 0, msec = 0, grn = 0, srn = 0;
    double x = 0., z = 0.;
    if      (sp->depfixtype == 8) strcpy(depfix, "A"); /* analyst */
    else if (sp->depfixtype == 1) strcpy(depfix, "B"); /* beyond limit */
    else if (sp->depfixtype == 2) strcpy(depfix, "H"); /* hypocentre */
    else if (sp->depfixtype == 3) strcpy(depfix, "D"); /* depdp */
    else if (sp->depfixtype == 4) strcpy(depfix, "S"); /* surface */
    else if (sp->depfixtype == 5) strcpy(depfix, "G"); /* depth grid */
    else if (sp->depfixtype == 6) strcpy(depfix, "M"); /* median depth */
    else if (sp->depfixtype == 7) strcpy(depfix, "R"); /* GRN depth */
    else                          strcpy(depfix, "F"); /* free depth */
/*
 *  origin time: day (seconds) and msec (milliseconds)
 *     note that sprintf does the rounding
 *     negative epoch times are handled correctly
 */
    sprintf(s, "%.3f\n", sp->time);
    x = atof(s);
    day = (int)x;
    sprintf(s, "%.3f", x - day);
    z = atof(s);
    if (z < 0.) {
        z += 1.;
        day--;
    }
    msec = (int)(1000. * z);
/*
 *  Flinn-Engdahl grn and srn
 */
    grn = gregnum(sp->lat, sp->lon, fep);
    srn = gregtosreg(grn);
/*
 *  previous prime is not ISC; get a new hypid
 */
    if (ep->isc_hypid == NULLVAL) {
        if (get_id("hypid", &nextid))
            return 1;
        if (verbose > 2)
            fprintf(logfp, "            new hypid=%d\n", nextid);
    }
/*
 *  remove prime flag from previous prime
 */
    sprintf(sql, "UPDATE %shypocenter SET prime = NULL \
                   WHERE isc_evid = %d",
            outdb, ep->evid);
    if (verbose > 2)
        fprintf(logfp, "            put_hypocenter: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("put_hypocenter: ");
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK ) {
        sprintf(errmsg, "put_hypocenter: %d", PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  delete previous prime if it was ISC
 */
    if (!nextid) {
        if (streq(indb, outdb)) {
            delete_isc_hypid(ep->isc_hypid);
        }
        else if (ep->outdbisc_hypid) {
            delete_isc_hypid(ep->outdbisc_hypid);
        }
    }
    else {
        ep->isc_hypid = nextid;
    }
/*
 *  populate hypocenter table
 */
    sprintf(psql, "INSERT INTO %shypocenter                                \
                   (hypid, pref_hypid, isc_evid, day, msec, lat, lon, srn, \
                    grn, depth, depdp, ndp, depfix, epifix, timfix, ndef,  \
                    nass, nsta, ndefsta, nrank, mindist, maxdist, azimgap, \
                    reporter, author, prime, etype, lddate) ", outdb);
    sprintf(psql, "%s VALUES (%d, %d, %d, ",
            psql, ep->isc_hypid, ep->isc_hypid, ep->evid);
    sprintf(psql, "%s TIMESTAMP 'epoch' + %d * INTERVAL '1 second', %d,", 
            psql, day, msec);
    sprintf(psql, "%s %f, %f, %d, %d, ", psql, sp->lat, sp->lon, srn, grn);
    if (sp->depdp == NULLVAL)
        sprintf(psql, "%s %f, NULL, NULL, ", psql, sp->depth);
    else
        sprintf(psql, "%s %f, %f, %d, ", psql, sp->depth, sp->depdp, sp->ndp);
/*
 *  depfix for free depth solutions is set to NULL
 */
    if (streq(depfix, "F")) sprintf(psql, "%s NULL, ", psql);
    else                    sprintf(psql, "%s '%s', ", psql, depfix);
    if (sp->epifix) sprintf(psql, "%s 'F', ", psql);
    else            sprintf(psql, "%s NULL, ", psql);
    if (sp->timfix) sprintf(psql, "%s 'F', ", psql);
    else            sprintf(psql, "%s NULL, ", psql);
    sprintf(psql, "%s %d, %d, %d, %d, %d, ",
            psql, sp->ndef, sp->nass, sp->nreading, sp->ndefsta, sp->prank);
    if (sp->mindist == NULLVAL)
        sprintf(psql, "%s NULL, NULL, NULL,", psql);
    else
        sprintf(psql, "%s %f, %f, %f,",
                psql, sp->mindist, sp->maxdist, sp->azimgap);
    sprintf(psql, "%s %d, '%s', 'P', '%s', NOW())",
            psql, repid, out_agency, ep->etype);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            put_hypocenter: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("put_hypocenter:");
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "put_hypocenter: %d", PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
    if (verbose) fprintf(logfp, "    put_hypocenter: done\n");
    return 0;
}

/*
 *  Title:
 *     put_hypoc_err
 *  Synopsis:
 *     Populates hypoc_err table.
 *  Input Arguments:
 *     ep  - pointer to event info
 *     sp  - pointer to current solution
 *  Called by:
 *     put_data
 *  Calls:
 *     pgsql_error, dropspace
 */
void put_hypoc_err(EVREC *ep, SOLREC *sp)
{
    extern double confidence;                         /* from config file */
    PGresult *res_set = (PGresult *)NULL;
    int i;
    char sql[2048], psql[2048], errmsg[2048];
    char conf[5];
    if      (fabs(confidence - 90.) < DEPSILON) strcpy(conf, "90");
    else if (fabs(confidence - 95.) < DEPSILON) strcpy(conf, "95");
    else if (fabs(confidence - 98.) < DEPSILON) strcpy(conf, "98");
    else                                        strcpy(conf, "NULL");
    sprintf(psql, "INSERT INTO %shypoc_err                                    \
                   (hypid, stt, stx, sty, stz, sxx, sxy, syy, syz, szx, szz,  \
                    sdobs, smajax, sminax, strike, stime, slon, slat, sdepth, \
                    sdepdp, reporter, author, confidence, lddate) VALUES (%d, ",
                outdb, ep->isc_hypid);
/*
 *  stt
 */
    if (sp->timfix)
        sprintf(psql, "%s NULL, NULL, NULL, NULL, ",psql);
    else {
        if (sp->covar[0][0] > 9999.99)
            sprintf(psql, "%s %f, ", psql, 9999.999);
        else
            sprintf(psql, "%s %f, ", psql, sp->covar[0][0]);
/*
 *      stx, sty
 */
        if (sp->epifix)
            sprintf(psql, "%s NULL, NULL, ",psql);
        else {
            if (fabs(sp->covar[0][1]) > 9999.99)
                sprintf(psql, "%s %f, ", psql, 9999.999);
            else
                sprintf(psql, "%s %f, ", psql, sp->covar[0][1]);
            if (fabs(sp->covar[0][2]) > 9999.99)
                sprintf(psql, "%s %f, ", psql, 9999.999);
            else
                sprintf(psql, "%s %f, ", psql, sp->covar[0][2]);
        }
/*
 *      stz
 */
        if (sp->depfix)
            sprintf(psql, "%s NULL, ",psql);
        else {
            if (fabs(sp->covar[0][3]) > 9999.99)
                sprintf(psql, "%s %f, ", psql, 9999.999);
            else
                sprintf(psql, "%s %f, ", psql, sp->covar[0][3]);
        }
    }
/*
 *  sxx, sxy, syy
 */
    if (sp->epifix)
        sprintf(psql, "%s NULL, NULL, NULL, NULL, NULL, ",psql);
    else {
        if (sp->covar[1][1] > 9999.99)
            sprintf(psql, "%s %f, ", psql, 9999.999);
        else
            sprintf(psql, "%s %f, ", psql, sp->covar[1][1]);
        if (fabs(sp->covar[1][2]) > 9999.99)
            sprintf(psql, "%s %f, ", psql, 9999.999);
        else
            sprintf(psql, "%s %f, ", psql, sp->covar[1][2]);
        if (fabs(sp->covar[2][2]) > 9999.99)
            sprintf(psql, "%s %f, ", psql, 9999.999);
        else
            sprintf(psql, "%s %f, ", psql, sp->covar[2][2]);
/*
 *      szx, syz
 */
        if (sp->depfix)
            sprintf(psql, "%s NULL, NULL, ",psql);
        else {
            if (fabs(sp->covar[1][3]) > 9999.99)
                sprintf(psql, "%s %f, ", psql, 9999.999);
            else
                sprintf(psql, "%s %f, ", psql, sp->covar[1][3]);
            if (fabs(sp->covar[2][3]) > 9999.99)
                sprintf(psql, "%s %f, ", psql, 9999.999);
            else
                sprintf(psql, "%s %f, ", psql, sp->covar[2][3]);
        }
    }
/*
 *  szz
 */
    if (sp->depfix)
        sprintf(psql, "%s NULL, ",psql);
    else {
        if (sp->covar[3][3] > 9999.99)
            sprintf(psql, "%s %f, ", psql, 9999.999);
        else
            sprintf(psql, "%s %f, ", psql, sp->covar[3][3]);
    }
/*
 *  uncertainties
 */
    if (sp->sdobs == NULLVAL)
        sprintf(psql, "%s NULL, ", psql);
    else if (sp->sdobs > 9999.99)
        sprintf(psql, "%s %f, ", psql, 9999.999);
    else
        sprintf(psql, "%s %f, ", psql, sp->sdobs);
    if (sp->smajax == NULLVAL)
        sprintf(psql, "%s NULL, NULL, NULL, ", psql);
    else {
        if (sp->sminax > 9999.99)
            sprintf(psql, "%s %f, %f, %f, ",
                    psql, 9999.999, 9999.999, sp->strike);
        else if (sp->smajax > 9999.99)
            sprintf(psql, "%s %f, %f, %f, ",
                    psql, 9999.999, sp->sminax, sp->strike);
        else
            sprintf(psql, "%s %f, %f, %f, ",
                    psql, sp->smajax, sp->sminax, sp->strike);
    }
    for (i = 0; i < 4; i++) {
        if (sp->error[i] == NULLVAL)
            sprintf(psql, "%s NULL, ", psql);
        else if (fabs(sp->error[i]) > 999.99)
            sprintf(psql, "%s %f, ", psql, 999.999);
        else
            sprintf(psql, "%s %f, ", psql, sp->error[i]);
    }
    if (sp->depdp_error == NULLVAL)
        sprintf(psql, "%s NULL, ", psql);
    else
        sprintf(psql, "%s %f, ", psql, sp->depdp_error);
/*
 *  reporter, author, confidence level
 */
    if (streq(conf, "NULL"))
        sprintf(psql, "%s %d, '%s', NULL, NOW())",
                psql, repid, out_agency);
    else
        sprintf(psql, "%s %d, '%s', '%s', NOW())",
                psql, repid, out_agency, conf);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp,"            put_hypoc_err: %s\n",sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("put_hypoc_err:");
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "put_hypoc_err: %d", PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
}

/*
 *  Title:
 *     put_hypoc_acc
 *  Synopsis:
 *     Populates hypoc_acc and network_quality tables.
 *  Input Arguments:
 *     ep  - pointer to event info
 *     sp  - pointer to current solution
 *     hq  - pointer to hypocentre quality structure
 *  Called by:
 *     put_data
 *  Calls:
 *     pgsql_error, dropspace
 */
void put_hypoc_acc(EVREC *ep, SOLREC *sp, HYPQUAL *hq)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[2048], psql[2048], errmsg[2048];
/*
 *  populate hypoc_acc table
 */
    sprintf(psql, "INSERT INTO %shypoc_acc                           \
                   (hypid, reporter, nstaloc, nsta10, gtcand, score) \
                   VALUES (%d, %d, %d, %d, %d, %f)",
            outdb, ep->isc_hypid, repid, sp->ndefsta,
            hq->ndefsta_10km, hq->gtcand, hq->score);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            put_hypoc_acc: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("put_hypoc_acc:");
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "put_hypoc_acc: %d", PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  populate network_quality table
 */
    if (hq->local_net.maxdist > 0.) {
        sprintf(psql, "INSERT INTO %snetwork_quality           \
                       (hypid, reporter, type, du, gap,        \
                        secondary_gap, nsta, mindist, maxdist) \
                       VALUES (%d, %d, 'local', %f, %f, %f, %d, %f, %f)",
                outdb, ep->isc_hypid, repid, hq->local_net.du,
                hq->local_net.gap, hq->local_net.sgap,
                hq->local_net.ndefsta, hq->local_net.mindist,
                hq->local_net.maxdist);
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            put_hyp: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_hyp: network_quality:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_hyp: network_quality: %d",
                    PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
    if (hq->near_net.maxdist > 0.) {
        sprintf(psql, "INSERT INTO %snetwork_quality           \
                       (hypid, reporter, type, du, gap,        \
                        secondary_gap, nsta, mindist, maxdist) \
                       VALUES (%d, %d, 'near', %f, %f, %f, %d, %f, %f)",
                outdb, ep->isc_hypid, repid, hq->near_net.du,
                hq->near_net.gap, hq->near_net.sgap,
                hq->near_net.ndefsta, hq->near_net.mindist,
                hq->near_net.maxdist);
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            put_hyp: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_hyp: network_quality:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_hyp: network_quality: %d",
                    PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
    if (hq->tele_net.maxdist > 0.) {
        sprintf(psql, "INSERT INTO %snetwork_quality           \
                       (hypid, reporter, type, du, gap,        \
                        secondary_gap, nsta, mindist, maxdist) \
                       VALUES (%d, %d, 'tele', %f, %f, %f, %d, %f, %f)",
                outdb, ep->isc_hypid, repid, hq->tele_net.du,
                hq->tele_net.gap, hq->tele_net.sgap,
                hq->tele_net.ndefsta, hq->tele_net.mindist,
                hq->tele_net.maxdist);
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            put_hyp: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_hyp: network_quality:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_hyp: network_quality: %d",
                    PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
    if (hq->whole_net.maxdist > 0.) {
        sprintf(psql, "INSERT INTO %snetwork_quality           \
                       (hypid, reporter, type, du, gap,        \
                        secondary_gap, nsta, mindist, maxdist) \
                       VALUES (%d, %d, 'whole', %f, %f, %f, %d, %f, %f)",
                outdb, ep->isc_hypid, repid, hq->whole_net.du,
                hq->whole_net.gap, hq->whole_net.sgap,
                hq->whole_net.ndefsta, hq->whole_net.mindist,
                hq->whole_net.maxdist);
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            put_hyp: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_hyp: network_quality:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_hyp: network_quality: %d",
                    PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
}

/*
 *  Title:
 *     put_netmag
 *  Synopsis:
 *     Populates netmag table
 *  Input Arguments:
 *     sp - pointer to current solution
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     put_data
 *  Uses:
 *     pgsql_error, dropspace, getid
 */
int put_netmag(SOLREC *sp)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[2048], psql[2048], errmsg[2048];
    int nextid = 0;
/*
 *  MS
 */
    if (sp->surfmag != NULLVAL) {
        if (get_id("magid", &nextid))
            return 1;
        sp->ms_id = nextid;
        sprintf(psql, "INSERT INTO %snetmag                               \
                       (hypid, magid, magtype, magnitude, nsta, reporter, \
                        uncertainty, nagency, author, lddate) VALUES ", outdb);
        sprintf(psql, "%s (%d, %d, 'MS', %f, %d, %d, %f, %d, '%s', NOW())",
                psql, sp->hypid, sp->ms_id, sp->surfmag, sp->nsta_ms,
                repid, sp->surfmag_uncertainty, sp->nMsagency, out_agency);
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            put_netmag: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_netmag:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_netmag: %d",
                    PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
/*
 *  mb
 */
    if (sp->bodymag != NULLVAL) {
        if (get_id("magid", &nextid))
            return 1;
        sp->mb_id = nextid;
        sprintf(psql, "INSERT INTO %snetmag                               \
                       (hypid, magid, magtype, magnitude, nsta, reporter, \
                        uncertainty, nagency, author, lddate) VALUES ", outdb);
        sprintf(psql, "%s (%d, %d, 'mb', %f, %d, %d, %f, %d, '%s', NOW())",
                psql, sp->hypid, sp->mb_id, sp->bodymag, sp->nsta_mb,
                repid, sp->bodymag_uncertainty, sp->nmbagency, out_agency);
        dropspace(psql, sql);
        if (verbose > 2)
            fprintf(logfp, "            put_netmag: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_netmag:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_netmag: %d",
                    PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
    return 0;
}

/*
 *  Title:
 *     put_stamag
 *  Synopsis:
 *     Populates stamag table.
 *     Deletes previous ISC stamag entries.
 *     Station magnitudes are written to the stamag table even if no network
 *        magnitude is calculated. In that case the magid field is set to null.
 *  Input Arguments:
 *     sp        - pointer to current solution
 *     stamag_mb - array of mb stamag structures
 *     stamag_ms - array of Ms stamag structures
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     put_data
 *  Uses:
 *     pgsql_error, dropspace, getid
 */
int put_stamag(SOLREC *sp, STAMAG *stamag_mb, STAMAG *stamag_ms)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[2048], psql[2048], errmsg[2048];
    int i, nextid = 0;
/*
 *  populate stamag table with station MS magnitudes
 */
    for (i = 0; i < sp->nsta_ms; i++) {
        if (get_id("stamag_stamagid_seq", &nextid))
            return 1;
        if (sp->ms_id == NULLVAL) {
            sprintf(psql, "INSERT INTO %sstamag                          \
                           (magid, hypid, magtype, magnitude, magdef,    \
                            sta, author, reporter, stamagid, lddate)     \
                           VALUES (NULL, %d, 'MS', %f, '%d', '%s', '%s', \
                                   %d, %d, NOW())",
                    outdb, sp->hypid,
                    stamag_ms[i].magnitude, stamag_ms[i].magdef,
                    stamag_ms[i].sta, out_agency, repid, nextid);
        }
        else {
            sprintf(psql, "INSERT INTO %sstamag                        \
                           (magid, hypid, magtype, magnitude, magdef,  \
                            sta, author, reporter, stamagid, lddate)   \
                           VALUES (%d, %d, 'MS', %f, '%d', '%s', '%s', \
                                   %d, %d, NOW())",
                    outdb, sp->ms_id, sp->hypid,
                    stamag_ms[i].magnitude, stamag_ms[i].magdef,
                    stamag_ms[i].sta, out_agency, repid, nextid);
        }
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            put_stamag: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_stamag:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_stamag: %d", PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
/*
 *  populate stamag table with station mb magnitudes
 */
    for (i = 0; i < sp->nsta_mb; i++) {
        if (get_id("stamag_stamagid_seq", &nextid))
            return 1;
        if (sp->mb_id == NULLVAL) {
            sprintf(psql, "INSERT INTO %sstamag                          \
                           (magid, hypid, magtype, magnitude, magdef,    \
                            sta, author, reporter, stamagid, lddate)     \
                           VALUES (NULL, %d, 'mb', %f, '%d', '%s', '%s', \
                                   %d, %d, NOW())",
                    outdb, sp->hypid,
                    stamag_mb[i].magnitude, stamag_mb[i].magdef,
                    stamag_mb[i].sta, out_agency, repid, nextid);
        }
        else {
            sprintf(psql, "INSERT INTO %sstamag                        \
                           (magid, hypid, magtype, magnitude, magdef,  \
                            sta, author, reporter, stamagid, lddate)   \
                           VALUES (%d, %d, 'mb', %f, '%d', '%s', '%s', \
                                   %d, %d, NOW())",
                     outdb, sp->mb_id, sp->hypid,
                     stamag_mb[i].magnitude, stamag_mb[i].magdef,
                     stamag_mb[i].sta, out_agency, repid, nextid);
        }
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            put_stamag: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_stamag:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_stamag: %d", PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
    return 0;
}

/*
 *  Title:
 *     put_rdmag
 *  Synopsis:
 *     Populates readingmag table.
 *     Deletes previous ISC readingmag entries.
 *     Reading magnitudes are written to the readingmag table even if no network
 *        magnitude is calculated. In that case the magid field is set to null.
 *  Input Arguments:
 *     sp        - pointer to current solution
 *     rdmag_mb  - array of mb rdmag structures
 *     rdmag_ms  - array of Ms rdmag structures
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     put_data
 *  Uses:
 *     pgsql_error, dropspace, getid
 */
int put_rdmag(SOLREC *sp, RDMAG *rdmag_mb, RDMAG *rdmag_ms)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[2048], psql[2048], errmsg[2048];
    int i, nextid = 0;
/*
 *  MS
 */
    for (i = 0; i < sp->nass_ms; i++) {
        if (get_id("rdmagid", &nextid))
            return 1;
        if (sp->ms_id == NULLVAL) {
            sprintf(psql, "INSERT INTO %sreadingmag              \
                    (magid, hypid, rdid, mtypeid, magnitude,     \
                     magdef, sta, author, repid, rdmagid) VALUES \
                    (NULL, %d, %d, %d, %f, %d, '%s', '%s', %d, %d)",
                    outdb, sp->hypid, rdmag_ms[i].rdid,
                    rdmag_ms[i].mtypeid, rdmag_ms[i].magnitude,
                    rdmag_ms[i].magdef, rdmag_ms[i].sta, out_agency,
                    repid, nextid);
        }
        else {
            sprintf(psql, "INSERT INTO %sreadingmag              \
                    (magid, hypid, rdid, mtypeid, magnitude,     \
                     magdef, sta, author, repid, rdmagid) VALUES \
                    (%d, %d, %d, %d, %f, %d, '%s', '%s', %d, %d)",
                    outdb, sp->ms_id, sp->hypid, rdmag_ms[i].rdid,
                    rdmag_ms[i].mtypeid, rdmag_ms[i].magnitude,
                    rdmag_ms[i].magdef, rdmag_ms[i].sta, out_agency,
                    repid, nextid);
        }
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            put_rdmag: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_rdmag:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_rdmag: %d", PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
/*
 *  mb
 */
    for (i = 0; i < sp->nass_mb; i++) {
        if (get_id("rdmagid", &nextid))
            return 1;
        if (sp->mb_id == NULLVAL) {
            sprintf(psql, "INSERT INTO %sreadingmag              \
                    (magid, hypid, rdid, mtypeid, magnitude,     \
                     magdef, sta, author, repid, rdmagid) VALUES \
                    (NULL, %d, %d, %d, %f, %d, '%s', '%s', %d, %d)",
                    outdb, sp->hypid, rdmag_mb[i].rdid,
                    rdmag_mb[i].mtypeid, rdmag_mb[i].magnitude,
                    rdmag_mb[i].magdef, rdmag_mb[i].sta, out_agency,
                    repid, nextid);
        }
        else {
            sprintf(psql, "INSERT INTO %sreadingmag              \
                    (magid, hypid, rdid, mtypeid, magnitude,     \
                     magdef, sta, author, repid, rdmagid) VALUES \
                    (%d, %d, %d, %d, %f, %d, '%s', '%s', %d, %d)",
                    outdb, sp->mb_id, sp->hypid, rdmag_mb[i].rdid,
                    rdmag_mb[i].mtypeid, rdmag_mb[i].magnitude,
                    rdmag_mb[i].magdef, rdmag_mb[i].sta, out_agency,
                    repid, nextid);
        }
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            put_rdmag: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_rdmag:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_rdmag: %d", PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
    return 0;
}

/*
 *  Title:
 *     delete_assoc
 *  Synopsis:
 *     Deletes previous ISC association records.
 *  Input Arguments:
 *     old_prime - hypid of previous prime hypocentre
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     put_data
 *  Calls:
 *     pgsql_error, dropspace
 */
static void delete_assoc(int old_prime, char *author)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[2048], psql[2048], errmsg[2048];
/*
 *  Delete ISC associations for this event.
 */
    sprintf(psql, "DELETE FROM %sassociation \
                   WHERE author = '%s' AND hypid = %d",
            outdb, author, old_prime);
    dropspace(psql, sql);
    if (verbose > 2) fprintf(logfp, "            delete_assoc: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("delete_assoc:");
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "delete_assoc: %d", PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
}

/*
 *  Title:
 *     put_assoc
 *  Synopsis:
 *     Populates association table.
 *  Input Arguments:
 *     sp  - pointer to current solution
 *     p[] - array of phase structures
 *  Called by:
 *     put_data
 *  Calls:
 *     pgsql_error, dropspace
 */
void put_assoc(SOLREC *sp, PHAREC p[])
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[2048], psql[2048], errmsg[2048];
    int i, j, k, n;
    char phase_fixed[2], phase[11], nondef[2];
    double timeres = 0., weight = 0.;
    int phase_fixed_ind, timeres_ind, nondef_ind;
/*
 *
 *  populate association table
 *
 */
    for (i = 0; i < sp->numphas; i++) {
        if (p[i].purged)       /* used by iscloc_search */
            continue;
        timeres_ind = 1;
        if      (p[i].resid == NULLVAL)  timeres_ind = 0;
        else if (isnan(p[i].resid))      timeres_ind = 0;
        else if (p[i].resid >= 1000000.)  timeres = 999999.;
        else if (p[i].resid <= -1000000.) timeres = -999999.;
        else                             timeres = p[i].resid;
/*
 *      preserve phase_fixed and nondef flags for the next pass
 */
        phase_fixed_ind = nondef_ind = 0;
        if (p[i].phase_fixed) {
            strcpy(phase_fixed, "F");
            phase_fixed_ind = 1;
        }
        if (p[i].force_undef) {
            strcpy(nondef, "U");
            nondef_ind = 1;
        }
/*
 *      insert association record
 */
        sprintf(psql, "INSERT INTO %sassociation                      \
                      (hypid, phid, phase, sta, delta, esaz, seaz,    \
                       timedef, weight, timeres, phase_fixed, nondef, \
                       deprecated, author, reporter, lddate)          \
                       VALUES (%d, %d, ",
                outdb, sp->hypid, p[i].phid);
/*
 *      fix for apostrophes in phasenames (e.g. P'P'df)
 */
        if (strstr(p[i].phase, "'")) {
            n = (int)strlen(p[i].phase);
            for (k = 0, j = 0; j < n; j++) {
                if (p[i].phase[j] == '\'')
                    phase[k++] = '\'';
                phase[k++] = p[i].phase[j];
            }
            phase[k] = '\0';
        }
        else
            strcpy(phase, p[i].phase);
        if (phase[0])        sprintf(psql, "%s '%s', ", psql, phase);
        else                 sprintf(psql, "%s NULL, ", psql);
        sprintf(psql, "%s '%s', %f, %f, %f, ",
                psql, p[i].sta, p[i].delta, p[i].esaz, p[i].seaz);
        if (p[i].timedef) {
            if (p[i].measerr == NULLVAL)
                sprintf(psql, "%s 'T', NULL, ", psql);
            else {
                weight = 1. / p[i].measerr;
                sprintf(psql, "%s 'T', %f, ", psql, weight);
            }
        }
        else                 sprintf(psql, "%s NULL, NULL,", psql);
        if (timeres_ind)     sprintf(psql, "%s %f, ", psql, timeres);
        else                 sprintf(psql, "%s NULL, ", psql);
        if (phase_fixed_ind) sprintf(psql, "%s '%s', ", psql, phase_fixed);
        else                 sprintf(psql ,"%s NULL, ", psql);
        if (nondef_ind)      sprintf(psql, "%s '%s', ", psql, nondef);
        else                 sprintf(psql ,"%s NULL, ", psql);
        sprintf(psql, "%s NULL, '%s', %d, NOW())",
                psql, out_agency, p[i].repid);
        dropspace(psql, sql);
        if (verbose > 2) fprintf(logfp, "            put_assoc: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_assoc:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_assoc: %d", PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
}

/*
 *  Title:
 *     put_ampmag
 *  Synopsis:
 *     Populates ampmag table.
 *     Deletes previous ampmag entries.
 *     Amplitude magnitudes are written to the ampmag table even if no network
 *        magnitude is calculated.
 *  Input Arguments:
 *     sp  - pointer to current solution
 *     p[] - array of phase structures
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     put_data
 *  Calls:
 *     pgsql_error, dropspace
 */
int put_ampmag(SOLREC *sp, PHAREC p[])
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[2048], psql[2048], errmsg[2048];
    int i, j, nextid = 0;
/*
 *  populate ampmag table
 */
    for (i = 0; i < sp->numphas; i++) {
        for (j = 0; j < p[i].numamps; j++) {
            if (!p[i].a[j].mtypeid) continue;
            if (get_id("ampmagid", &nextid))
                return 1;
            sprintf(psql, "INSERT INTO %sampmag                    \
                           (ampmagid, hypid, rdid, ampid, mtypeid, \
                           magnitude, ampdef, repid, author)       \
                           VALUES (%d, %d, %d, %d, %d, %f, %d, %d, '%s')",
                    outdb, nextid, sp->hypid, p[i].rdid, p[i].a[j].ampid,
                    p[i].a[j].mtypeid, p[i].a[j].magnitude, p[i].a[j].ampdef,
                    repid, out_agency);
            dropspace(psql, sql);
            if (verbose > 2)
                fprintf(logfp, "            put_ampmag: %s\n", sql);
            if ((res_set = PQexec(conn, sql)) == NULL)
                pgsql_error("put_ampmag:");
            else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
                sprintf(errmsg, "put_ampmag: %d", PQresultStatus(res_set));
                pgsql_error(errmsg);
            }
            PQclear(res_set);
        }
    }
    return 0;
}

/*
 *  Title:
 *     put_mszh
 *  Synopsis:
 *     Populates ms_zh table.
 *     Deletes previous ms_zh entries.
 *     MS_Z, MS_H magnitudes are written to the ms_zh table even if no network
 *        MS magnitude is calculated.
 *  Input Arguments:
 *     sp   - pointer to current solution
 *     p[]  - array of phase structures
 *     mszh - array to MS vertical/horizontal magnitude structures
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     put_data
 *  Calls:
 *     pgsql_error, dropspace
 */
int put_mszh(SOLREC *sp, MSZH *mszh)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[2048], psql[2048], errmsg[2048];
    int i, nextid = 0;
/*
 *  populate ms_zh table
 */
    for (i = 0; i < sp->nass_ms; i++) {
        if (get_id("mszhid", &nextid))
            return 1;
        sprintf(psql, "INSERT INTO %sms_zh                \
                       (mszhid, hypid, rdid, msz, mszdef, \
                        msh, mshdef,repid, author)        \
                       VALUES (%d, %d, %d, ",
                outdb, nextid, sp->hypid, mszh[i].rdid);
        if (mszh[i].msz == NULLVAL)
            sprintf(psql, "%s NULL, %d, ", psql, mszh[i].mszdef);
        else
            sprintf(psql, "%s %f, %d, ", psql, mszh[i].msz, mszh[i].mszdef);
        if (mszh[i].msh == NULLVAL)
            sprintf(psql, "%s NULL, %d, ", psql, mszh[i].mshdef);
        else
            sprintf(psql, "%s %f, %d, ", psql, mszh[i].msh, mszh[i].mshdef);
        sprintf(psql, "%s %d, '%s')", psql, repid, out_agency);
        dropspace(psql, sql);
        if (verbose > 2)
            fprintf(logfp, "            put_mszh: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("put_mszh:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "put_mszh: %d", PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
    return 0;
}


/*
 *  Title:
 *     remove_isc
 *  Synopsis:
 *     Removes existing ISC hypocentre from database if locator failed to get
 *     a convergent solution.
 *  Input Arguments:
 *     ep - pointer to event info
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     eventloc
 *  Calls:
 *     pgsql_error
 *  Notes:
 *     Need to run replace_prime() and replace_assoc() after this.
 */
void remove_isc(EVREC *ep)
{
    char sql[1024];
    PGresult *res_set = (PGresult *)NULL;
    int hypid = 0;
    if (streq(indb, outdb))
        hypid = ep->isc_hypid;
    else
        hypid = ep->outdbisc_hypid;
    if (hypid) {
        if (verbose) fprintf(logfp, "    remove_isc: %d\n", hypid);
        delete_isc_hypid(hypid);
        sprintf(sql, "DELETE FROM %spub_comments \
                      WHERE id_name = 'hypid' AND id_value = %d",
                outdb, hypid);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("remove_isc: pub_comments:");
        PQclear(res_set);
        if (verbose)
            fprintf(logfp, "    remove_isc: hypocentre removed.\n");
    }
}

/*
 *  Title:
 *     delete_isc_hypid
 *  Synopsis:
 *     Removes existing ISC hypocentre from database.
 *  Input Arguments:
 *     isc_hypid - ISC hypid
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     put_hypocenter, remove_isc
 *  Calls:
 *     pgsql_error
 */
static void delete_isc_hypid(int isc_hypid)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[1024], errmsg[1024];
/*
 *  delete hypid from hypocenter
 */
    sprintf(sql, "DELETE FROM %shypocenter WHERE hypid = %d",
            outdb, isc_hypid);
    if ((res_set = PQexec(conn, sql)) == NULL) {
        pgsql_error("delete_isc_hypid: hypoc:");
    }
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "delete_isc_hypid: hypoc: %d",
                PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  delete hypid from hypoc_err
 */
    sprintf(sql, "DELETE FROM %shypoc_err WHERE hypid = %d",
            outdb, isc_hypid);
    if ((res_set = PQexec(conn, sql)) == NULL) {
        pgsql_error("delete_isc_hypid: hypoc_err:");
    }
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "delete_isc_hypid: hypoc_err: %d",
               PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  delete hypid from hypoc_acc
 */
    sprintf(sql, "DELETE FROM %shypoc_acc WHERE hypid = %d",
            outdb, isc_hypid);
    if ((res_set = PQexec(conn, sql)) == NULL) {
        pgsql_error("delete_isc_hypid: hypoc_acc:");
    }
    else if ( PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "delete_isc_hypid: hypoc_acc: %d",
                PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  delete hypid from network_quality
 */
    sprintf(sql, "DELETE FROM %snetwork_quality WHERE hypid = %d",
            outdb, isc_hypid);
    if ((res_set = PQexec(conn, sql)) == NULL) {
        pgsql_error("delete_isc_hypid: network_quality:");
    }
    else if ( PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "delete_isc_hypid: network_quality: %d",
                PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  delete hypid from netmag
 */
    sprintf(sql, "DELETE FROM %snetmag WHERE hypid = %d",
            outdb, isc_hypid);
    if ((res_set = PQexec(conn,sql)) == NULL) {
        pgsql_error("delete_isc_hypid: netmag:");
    }
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "delete_isc_hypid: netmag: %d",
                PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  delete hypid from stamag
 */
    sprintf(sql, "DELETE FROM %sstamag WHERE hypid = %d",
            outdb, isc_hypid);
    if ((res_set = PQexec(conn, sql)) == NULL) {
        pgsql_error("delete_isc_hypid: stamag:");
    }
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "delete_isc_hypid: stamag: %d",
                PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  delete hypid from readingmag
 */
    sprintf(sql, "DELETE FROM %sreadingmag WHERE hypid = %d",
            outdb, isc_hypid);
    if ((res_set = PQexec(conn, sql)) == NULL) {
        pgsql_error("delete_isc_hypid: readingmag:");
    }
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "delete_isc_hypid: readingmag: %d",
                PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  delete hypid from ampmag
 */
    sprintf(sql, "DELETE FROM %sampmag WHERE hypid = %d",
            outdb, isc_hypid);
    if ((res_set = PQexec(conn, sql)) == NULL) {
        pgsql_error("delete_isc_hypid: ampmag:");
    }
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "delete_isc_hypid: ampmag: %d",
                PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  delete hypid from ms_zh
 */
    sprintf(sql, "DELETE FROM %sms_zh WHERE hypid = %d",
            outdb, isc_hypid);
    if ((res_set = PQexec(conn, sql)) == NULL) {
        pgsql_error("delete_isc_hypid: ms_zh:");
    }
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "delete_isc_hypid: ms_zh: %d",
                PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
    if (verbose)
        fprintf(logfp, "    delete_isc_hypid: %s hypocentre removed.\n",
                in_agency);
}

/*
 *  Title:
 *     replace_prime
 *  Synopsis:
 *     Changes prime hypocentre of event without removing old one.
 *  Input Arguments:
 *     ep  - pointer to event info
 *     hp  - pointer to preferred hypocentre
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     eventloc, synthetic
 *  Calls:
 *     pgsql_error
 */
void replace_prime(EVREC *ep, HYPREC *hp)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[1536], errmsg[1536];
/*
 *  if event is not in outdb, put it there
 */
    if (strcmp(indb, outdb) && !ep->outdbprime) {
        if (verbose) fprintf(logfp, "        put_event (%.2f)\n", secs(&t0));
        sprintf(sql, "INSERT INTO %sevent SELECT * from %sevent \
                       WHERE evid = %d", outdb, indb, ep->evid);
        if (verbose > 2) fprintf(logfp, "            replace_prime: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("replace_prime:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "replace_prime: %d", PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        sprintf(sql, "INSERT INTO %shypocenter SELECT * from %shypocenter \
                       WHERE hypid = %d", outdb, indb, hp->hypid);
        if (verbose > 2) fprintf(logfp, "            replace_prime: %s\n", sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("replace_prime:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "replace_prime: %d", PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
/*
 *  Update event table
 */
    sprintf(sql, "UPDATE %sevent SET prime_hyp = %d WHERE evid = %d",
            outdb, hp->hypid, ep->evid);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("replace_prime:");
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "replace_prime: %d", PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  Update old prime
 */
    sprintf(sql, "UPDATE %shypocenter SET prime = NULL WHERE isc_evid = %d",
            outdb, ep->evid);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("replace_prime: ");
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "replace_prime: %d", PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
/*
 *  Update new prime
 */
    sprintf(sql, "UPDATE %shypocenter SET prime = 'P' WHERE hypid = %d",
            outdb, hp->hypid);
    if ((res_set = PQexec(conn, sql)) == NULL)
        pgsql_error("replace_prime: ");
    else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "replace_prime: %d", PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
    if (verbose)
        fprintf(logfp, "    replace_prime: prime: %d evid: %d\n",
                hp->hypid, ep->evid);
}

/*
 *  Title:
 *     replace_assoc
 *  Synopsis:
 *     Rolls back associations to previous prime.
 *  Input Arguments:
 *     ep  - pointer to event info
 *     p[] - array of phase structures.
 *     hp  - pointer to prefered hypocentre
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     eventloc, synthetic
 *  Calls:
 *     pgsql_error
 */
void replace_assoc(EVREC *ep, PHAREC p[], HYPREC *hp)
{
    PGresult *res_set = (PGresult *)NULL;
    char sql[2048], errmsg[2048];
    int hypid = 0;
    char author[VALLEN];
    double timeres = 0.;
    char  phase_fixed[2], phase[11];
    short timeres_ind, phase_fixed_ind;
    int i, j, k, n;
/*
 *  delete previous assocs
 */
    strcpy(author, "");
    if (streq(indb, outdb)) {
        hypid = ep->prime;
        strcpy(author, in_agency);
    }
    else {
        hypid = ep->outdbprime;
        strcpy(author, out_agency);
    }
    if (!hypid)
        return;
/*
 *  Delete current associations and put back old associations
 */
    delete_assoc(hypid, author);
    if (verbose)
        fprintf(logfp, "    replace_assoc: insert associations to hypid=%d\n",
                hp->hypid);
    for (i = 0; i < ep->numphas; i++) {
/*
 *      Want phase_fixed flag to remain for next pass.
 */
        phase_fixed_ind = 0;
        if (p[i].phase_fixed) {
            strcpy(phase_fixed, "F");
            phase_fixed_ind = 1;
        }
        timeres_ind = 1;
        if (p[i].resid == NULLVAL)       timeres_ind = 0;
        else if (isnan(p[i].resid))      timeres_ind = 0;
        else if (p[i].resid >= 1000000)  timeres = 999999;
        else if (p[i].resid <= -1000000) timeres = -999999;
        else                             timeres = p[i].resid;
/*
 *      populate association table
 */
        sprintf(sql,"INSERT INTO %sassociation \
                    (hypid, phid, phase, sta, delta, esaz, seaz,     \
                     timeres, phase_fixed, author, reporter, lddate) \
                     VALUES (%d, %d,",
                outdb, hp->hypid, p[i].phid);
/*
 *      fix for apostrophes in phasenames (e.g. P'P'df)
 */
        if (strstr(p[i].phase, "'")) {
            n = (int)strlen(p[i].phase);
            for (k = 0, j = 0; j < n; j++) {
                if (p[i].phase[j] == '\'')
                    phase[k++] = '\'';
                phase[k++] = p[i].phase[j];
            }
            phase[k] = '\0';
        }
        else
            strcpy(phase, p[i].phase);
        if (phase[0]) sprintf(sql, "%s '%s', ", sql, phase);
        else          sprintf(sql, "%s NULL, ", sql);
        sprintf(sql, "%s '%s', %f, %f, %f,",
                sql, p[i].sta, p[i].delta, p[i].esaz, p[i].seaz);
        if (timeres_ind) sprintf(sql, "%s %f,", sql, timeres);
        else             sprintf(sql, "%s NULL,", sql);
        if (phase_fixed_ind) sprintf(sql, "%s '%s',", sql, phase_fixed);
        else                 sprintf(sql ,"%s NULL,", sql);
        sprintf(sql, "%s '%s', %d, NOW())", sql, out_agency, p[i].repid);
        if (verbose > 2)
            fprintf(logfp, "            replace_assoc: %4d: %s\n", i, sql);
        if ((res_set = PQexec(conn, sql)) == NULL)
            pgsql_error("replace_assoc:");
        else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
            sprintf(errmsg, "replace_assoc: %d", PQresultStatus(res_set));
            pgsql_error(errmsg);
        }
        PQclear(res_set);
    }
}

/*
 *  Title:
 *      get_id
 *  Synopsis:
 *      Gets next unique id from DB sequence in the isc account.
 *  Input Arguments:
 *      sequence - sequence name
 *  Output Arguments:
 *      nextid   - unique id
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     put_hypocenter, put_netmag, put_stamag, put_rdmag, put_ampmag, put_ms_zh
 *  Calls:
 *     pgsql_error
 *
 */
int get_id(char *sequence, int *nextid)
{
    extern char nextid_db[24];             /* get new ids from this account */
    PGresult *res_set = (PGresult *)NULL;
    char sql[1024];
    sprintf(sql, "SELECT NEXTVAL('%s.%s')", nextid_db, sequence);
    if (verbose > 2) fprintf(logfp, "            get_id: %s\n", sql);
    if ((res_set = PQexec(conn, sql)) == NULL) {
        pgsql_error("get_id: error: nextid:");
        return 1;
    }
    else if (PQntuples(res_set) == 1) {
        *nextid = atoi(PQgetvalue(res_set, 0, 0));
    }
    PQclear(res_set);
    if (verbose > 2) fprintf(logfp, "            new id=%d\n", *nextid);
    return 0;
}

#endif /* WITH_DB */

/*  EOF  */
