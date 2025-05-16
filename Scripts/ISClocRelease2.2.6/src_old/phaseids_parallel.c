#include "iscloc.h"
extern int verbose;
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;
extern char no_resid_phase[MAXNUMPHA][PHALEN];        /* from model file */
extern int no_resid_phase_num;        /* number of 'non-residual' phases */
extern char allowable_phase[MAXTTPHA][PHALEN];        /* from model file */
extern int no_allowable_phase_num;         /* number of allowable phases */
extern PHASEMAP phase_map[];           /* phase mapping from model file */
extern int phase_map_num;   /* number of phase mappings from model file */
extern char firstPphase[MAXTTPHA][PHALEN];     /* first-arriving P list */
extern int firstPphase_num;        /* number of first-arriving P phases */
extern char firstPopt[MAXTTPHA][PHALEN];       /* optional first P list */
extern int firstPopt_num;          /* number of optional first P phases */
extern char firstSphase[MAXTTPHA][PHALEN];     /* first-arriving S list */
extern int firstSphase_num;        /* number of first-arriving S phases */
extern char firstSopt[MAXTTPHA][PHALEN];       /* optional first S list */
extern int firstSopt_num;          /* number of optional first S phases */
extern PHASEWEIGHT phase_weight[MAXNUMPHA];          /* from model file */
extern int phase_weight_num;    /* number of rows in phase weight table */
extern int do_correlated_errors;
extern double sigmathres;

/*
 * Functions:
 *    id_pha
 *    reidentify_pha
 *    id_pfake
 *    remove_pfake
 *    mark_duplicates
 *    reported_phase_resid
 */

/*
 * Local functions:
 *    phaseid
 *    isfirstP
 *    isfirstS
 *    setsigma
 *    same_sta
 *    same_time
 */
static void phaseid(SOLREC *sp, READING *rdindx, PHAREC p[], EC_COEF *ec,
                    TT_TABLE *tt_tables, short int **topo);
static int isfirstP(char *phase, char *mappedphase);
static int isfirstS(char *phase, char *mappedphase);
static void setsigma(PHAREC *pp);
static void same_sta(int samesta[], int n, SOLREC *sp, PHAREC p[], EC_COEF *ec,
                     TT_TABLE *tt_tables, short int **topo);
static void same_time(int sametime[], int n, SOLREC *sp, PHAREC p[],
                      EC_COEF *ec, TT_TABLE *tt_tables, short int **topo);

/*
 *  Title:
 *     id_pha
 *  Synopsis:
 *     Identifies phases with respect to the initial hypocentre.
 *     Maps reported phase ids to IASPEI standard phase names.
 *     Uses phase dependent information from <vmodel>_model.txt file.
 *        phase_map - list of possible reported phases with their
 *                    corresponding IASPEI phase id.
 *     Sets unrecognized reported phase names to null.
 *     Assumes that unidentified reported first-arriving phase is P.
 *     Identifies phases within a reading and marks first-arriving P and S.
 *     Sets time-defining flags and a priori measurement errors.
 *  Input Arguments:
 *     sp        - pointer to current solution
 *     rdindx    - array of reading structures
 *     p         - array of phase structures
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *  Output Arguments:
 *     p         - array of phase structures
 *  Returns:
 *     initial number of time defining phases
 *  Called by:
 *     eventloc, fixedhypo
 *  Calls:
 *     phaseid, setsigma
 */
int id_pha(SOLREC *sp, READING *rdindx, PHAREC p[], EC_COEF *ec,
           TT_TABLE *tt_tables, short int **topo)
{
    int i, j, n = 0, np;
/*
 *  map phases to IASPEI phase names
 */
    for (i = 0; i < sp->numphas; i++) {
/*
 *      initializations
 */
        strcpy(p[i].prevphase, "");
        p[i].dupsigma = 0.;
        p[i].timedef = p[i].prevtimedef = 0;
        p[i].firstP = p[i].firstS = 0;
        p[i].duplicate = 0;
/*
 *      continue if phase name is fixed by analysts
 */
        if (p[i].phase_fixed)
            continue;
/*
 *      assume that unknown initial phases are P
 */
        strcpy(p[i].phase, "");
        if (streq(p[i].rep_phase, "") && p[i].init)
            strcpy(p[i].phase, "P");
/*
 *      map reported phase names to IASPEI standard
 */
        for (j = 0; j < phase_map_num; j++) {
            if (streq(p[i].rep_phase, phase_map[j].rep_phase)) {
                strcpy(p[i].phase, phase_map[j].phase);
                break;
            }
        }
    }
/*
 *  identify first arriving P and S in a reading
 */
    for (i = 0; i < sp->nreading; i++) {
        np = rdindx[i].start + rdindx[i].npha;
        for (j = rdindx[i].start; j < np; j++) {
            if (p[j].phase[0] == 'P' ||
                (islower(p[j].phase[0]) &&
                 (p[j].phase[1] == 'P' || p[j].phase[1] == 'w'))) {
                p[j].firstP = 1;
                break;
            }
        }
        for (j = rdindx[i].start; j < np; j++) {
            if (p[j].phase[0] == 'S' ||
                (islower(p[j].phase[0]) && p[j].phase[1] == 'S')) {
                p[j].firstS = 1;
                break;
            }
        }
    }
/*
 *  identify phases within a reading
 */
#ifdef WITH_GCD
/*
 *  use GCD (Mac OS) to parallelize the phase identification process
 *  each reading is processed concurrently
 */
    dispatch_apply(sp->nreading, dispatch_get_global_queue(0, 0), ^(size_t i){
        phaseid(sp, &rdindx[i], p, ec, tt_tables, topo);
    });
#else
/*
 *  single core
 */
    for (i = 0; i < sp->nreading; i++) {
        phaseid(sp, (rdindx + i), p, ec, tt_tables, topo);
    }
#endif
/*
 *  set timedef flags and get prior measurement errors
 */
    for (n = 0, i = 0; i < sp->numphas; i++) {
        if (strcmp(p[i].phase, p[i].prevphase)) {
            setsigma(&p[i]);
            if (verbose > 3)
                fprintf(logfp, "        %-6s %-8s: timedef=%d deltim=%.3f\n",
                        p[i].sta, p[i].phase, p[i].timedef, p[i].measerr);
        }
        if (p[i].measerr == NULLVAL) p[i].timedef = 0;
        if (p[i].timedef) n++;
    }
    return n;
}

/*
 *  Title:
 *     reidentify_pha
 *  Synopsis:
 *     Reidentifies phases after NA search is completed and/or
 *         if depth crosses Moho/Conrad discontinuity between iterations.
 *     At this point phase names are already mapped to IASPEI standards.
 *     Identifies phases within a reading and marks first-arriving P and S.
 *     Sets time-defining flags and a priori measurement errors.
 *  Input Arguments:
 *     sp        - pointer to current solution
 *     rdindx   - array of reading structures
 *     p         - array of phase structures
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *  Output Arguments:
 *     p         - array of phase structures
 *  Return:
 *     1 on phasename changes, 0 otherwise
 *  Called by:
 *     eventloc, locate_event
 *  Calls:
 *     phaseid, setsigma
 */
int reidentify_pha(SOLREC *sp, READING *rdindx, PHAREC p[], EC_COEF *ec,
                   TT_TABLE *tt_tables, short int **topo)
{
    int i, j, np, isphasechange = 0;
    for (i = 0; i < sp->numphas; i++)
        p[i].firstP = p[i].firstS = p[i].duplicate = 0;
/*
 *  identify first arriving P and S in a reading
 */
    for (i = 0; i < sp->nreading; i++) {
        np = rdindx[i].start + rdindx[i].npha;
        for (j = rdindx[i].start; j < np; j++) {
            if (p[j].phase[0] == 'P' ||
                (islower(p[j].phase[0]) &&
                 (p[j].phase[1] == 'P' || p[j].phase[1] == 'w'))) {
                p[j].firstP = 1;
                break;
            }
        }
        for (j = rdindx[i].start; j < np; j++) {
            if (p[j].phase[0] == 'S' ||
                (islower(p[j].phase[0]) && p[j].phase[1] == 'S')) {
                p[j].firstS = 1;
                break;
            }
        }
    }
/*
 *  identify phases within a reading
 */
#ifdef WITH_GCD
/*
 *  use GCD (Mac OS) to parallelize the phase identification process
 *  each reading is processed concurrently
 */
    dispatch_apply(sp->nreading, dispatch_get_global_queue(0, 0), ^(size_t i){
        phaseid(sp, &rdindx[i], p, ec, tt_tables, topo);
    });
#else
/*
 *  single core
 */
    for (i = 0; i < sp->nreading; i++) {
        phaseid(sp, &rdindx[i], p, ec, tt_tables, topo);
    }
#endif
/*
 *  set timedef flags and get prior measurement errors
 */
    for (i = 0; i < sp->numphas; i++) {
        if (strcmp(p[i].phase, p[i].prevphase)) {
            setsigma(&p[i]);
            isphasechange = 1;
            if (verbose > 2) {
                fprintf(logfp, "        %-6s %-8s -> %-8s: timedef=%d ",
                        p[i].sta, p[i].prevphase, p[i].phase, p[i].timedef);
                fprintf(logfp, "deltim=%.3f\n", p[i].measerr);
            }
            strcpy(p[i].prevphase, p[i].phase);
        }
        if (p[i].measerr == NULLVAL) p[i].timedef = 0;
   }
    return isphasechange;
}

/*
 *  Title:
 *     phaseid
 *  Synopsis:
 *     Identifies phases in a reading according to their time residuals.
 *     At this point phase names are already mapped to IASPEI standards.
 *     Uses phase dependent information from <vmodel>_model.txt file.
 *        no_resid_phase - list of phases that don't get residuals, i.e.
 *                         never used in the location (e.g. amplitude phases)
 *        allowable_phases - list of phases to which reported phases can be
 *                         renamed
 *        allowable_first_P - list of phases to which reported first-arriving
 *                         P phases can be renamed
 *        optional_first_P - additional list of phases to which reported
 *                         first-arriving P phases can be renamed
 *        allowable_first_S - list of phases to which reported first-arriving
 *                         S phases can be renamed
 *        optional_first_S - additional list of phases to which reported
 *                         first-arriving S phases can be renamed
 *     Skips phases that do not get residuals.
 *        The list of no-residual phases contains the list of phases for which
 *        residuals will not be calculated, such as amplitude phases or IASPEI
 *        phases for which no travel-time tables exist. These phases are not
 *        used in the location.
 *     Considers only P or S type phases.
 *        phase type is determined by the first leg of the phase id;
 *        for depth phases phase type is determined by the second letter.
 *     Does not reidentify phases fixed by analyst (phase_fixed flag).
 *     For exact duplicates (|dT| < 0.01) keeps the first non-null phaseid.
 *     Checks if the reported phase is in the list of allowable phases.
 *        The list of allowable phases were introduced to prevent the locator
 *        to rename phases to unlikely 'exotic' phases, just because a
 *        travel-time prediction fits better the observed travel-time. For
 *        instance, we do not want to reidentify a reported Sn as SgSg, SPn or
 *        horribile dictu, sSn. Recall that phases may suffer from picking
 *        errors or the observed travel-times may reflect true 3D structures
 *        not modeled by the velocity model. Introducing the list of
 *        allowable phases helps to maintain the sanity of the bulletin and
 *        mitigates the risk of misidentifying phases. However, if a reported
 *        phase is not in the list of allowable phases, it is temporarily
 *        added to the list accepting the fact that station operators may
 *        confidently pick later phases. In other words, exotic phase names
 *        can appear in the final bulletin only if they were reported as such.
 *     Loops through the (possibly amended) list of allowable phases and
 *        calculates the time residual.
 *        Does not allow renaming a P-type phase to S-type, and vice versa.
 *        Does not allow allow S(*) phases to be renamed to depth phases (s*).
 *        Does not allow for repeating phaseids in a reading.
 *        Further restrictions apply to first-arriving P and S phases.
 *            First-arriving P and S phases can be identified as those in the
 *            list of allowable first-arriving P and S phases. Occasionally a
 *            station operator may not report the true first-arriving phase
 *            due to high noise conditions. To account for this situation the
 *            list of optional first-arriving P and S phases is also checked.
 *        Keeps track of the phaseid with the smallest residual.
 *     Sets the ISC phase id to the phase in the allowable phase list with the
 *        smallest residual.
 *     If no eligible phase is found, leaves the phase unidentified.
 *  Input Arguments:
 *     sp        - pointer to current solution
 *     rdindx    - reading record
 *     pp        - phase record
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *  Output Arguments:
 *     p         - array of phase structures
 *  Called by:
 *     id_pha, reidentify_pha
 *  Calls:
 *     read_ttime, isfirstP, isfirstS
 */
static void phaseid(SOLREC *sp, READING *rdindx, PHAREC p[], EC_COEF *ec,
                    TT_TABLE *tt_tables, short int **topo)
{
    double resid, bigres = 60., min_resid, dtdd = NULLVAL;
    double ttime = NULLVAL, pPttime = NULLVAL;
    char candidate_phase[PHALEN], mappedphase[PHALEN], phase[PHALEN];
    int j, k, m, n, ii, isseen, isS, isP, iss, isp, ptype, stype, ttype;
    int npha;
/*
 *  loop over phases in this reading
 */
    n = rdindx->start + rdindx->npha;
    for (m = 0, k = rdindx->start; k < n; m++, k++) {
/*
 *      skip phases that don't get residuals (amplitudes etc)
 */
        for (j = 0; j < no_resid_phase_num; j++) {
            if (streq(p[k].phase, no_resid_phase[j]))
                break;
        }
        if (j != no_resid_phase_num)
            continue;
        min_resid = bigres + 1.;
        resid = NULLVAL;
/*
 *      phase type is determined by the first leg of the phase id
 */
        isS = isP = iss = isp = ptype = stype = 0;
        if (p[k].phase[0] == 'P') isP = 1;
        if (p[k].phase[0] == 'S') isS = 1;
/*
 *      for depth phases phase type is determined by the second letter
 */
        if (islower(p[k].phase[0])) {
            if (p[k].phase[1] == 'P' || p[k].phase[1] == 'w') isp = 1;
            if (p[k].phase[1] == 'S') iss = 1;
        }
/*
 *      consider only P or S-type phases
 */
        if (!(isP || isp || isS || iss)) continue;
        if (isP || isp) ptype = 1;
        if (isS || iss) stype = 1;
/*
 *      do not reidentify fixed phase names
 */
        if (p[k].phase_fixed) {
/*
 *          get travel time for phase
 */
            if (read_ttime(sp, &p[k], ec, tt_tables, topo, 0, -1)) {
                p[k].ttime = NULLVAL;
                p[k].resid = NULLVAL;
                if (verbose) {
                    fprintf(logfp, "    can't get TT for fixed phase %s! ",
                            p[k].phase);
                    fprintf(logfp, "(depth=%.2f delta=%.2f sta=%s)\n",
                            sp->depth, p[k].delta, p[k].sta);
                }
            }
            else {
                p[k].resid = p[k].time - sp->time - p[k].ttime;
#ifdef SERIAL
                if (verbose > 4)
                    fprintf(logfp, "              %-8s %9.4f %9.4f\n",
                            p[k].phase, p[k].ttime, p[k].resid);
#endif
            }
        }
/*
 *      phase is not fixed by analysts
 */
        else {
/*
 *          deal with duplicates: keep the first non-null phaseid
 *              phases ordered by time within a reading
 */
            if (m) {
                if (fabs(p[k].time - p[k-1].time) < 0.01) {
/*
 *                  if previous phaseid is null, rename it
 */
                    if (streq(p[k-1].phase, "")) {
                        strcpy(p[k-1].phase, p[k].phase);
                        p[k-1].ttime = p[k].ttime;
                        p[k-1].resid = p[k].resid;
                        p[k-1].dtdd = p[k].dtdd;
                        continue;
                    }
/*
 *                  otherwise use previous phase
 */
                    else {
                        for (j = 0; j < no_resid_phase_num; j++) {
                            if (streq(p[k-1].phase, no_resid_phase[j]))
                                break;
                        }
/*
 *                      if it is a no-residual phase, leave it alone
 */
                        if (j != no_resid_phase_num)
                            continue;
                        strcpy(p[k].phase, p[k-1].phase);
                        p[k].ttime = p[k-1].ttime;
                        p[k].resid = p[k-1].resid;
                        p[k].dtdd = p[k-1].dtdd;
                        continue;
                    }
                }
            }
            strcpy(mappedphase, p[k].phase);
/*
 *
 *          see if mapped phase is in the allowable phase list
 *
 */
            npha = no_allowable_phase_num;
            for (j = 0; j < no_allowable_phase_num; j++) {
                if (streq(mappedphase, allowable_phase[j]))
                    break;
            }
/*
 *          not in the list; temporarily add it to the list
 */
            if (j == no_allowable_phase_num) {
                strcpy(allowable_phase[npha++], mappedphase);
/*
 *              deal with PP et al (PP is in the list of allowable phases)
 */
                if (streq(mappedphase, "PnPn")) {
                    strcpy(allowable_phase[npha++], "PbPb");
                    strcpy(allowable_phase[npha++], "PgPg");
                }
                else if (streq(mappedphase, "PbPb")) {
                    strcpy(allowable_phase[npha++], "PnPn");
                    strcpy(allowable_phase[npha++], "PgPg");
                }
                else if (streq(mappedphase, "PgPg")) {
                    strcpy(allowable_phase[npha++], "PnPn");
                    strcpy(allowable_phase[npha++], "PbPb");
                }
/*
 *              deal with SS et al (SS is in the list of allowable phases)
 */
                if (streq(mappedphase, "SnSn")) {
                    strcpy(allowable_phase[npha++], "SbSb");
                    strcpy(allowable_phase[npha++], "SgSg");
                }
                else if (streq(mappedphase, "SbSb")) {
                    strcpy(allowable_phase[npha++], "SnSn");
                    strcpy(allowable_phase[npha++], "SgSg");
                }
                else if (streq(mappedphase, "SgSg")) {
                    strcpy(allowable_phase[npha++], "SnSn");
                    strcpy(allowable_phase[npha++], "SbSb");
                }
/*
 *              deal with PS et al
 */
                if (streq(mappedphase, "PS")) {
                    strcpy(allowable_phase[npha++], "PnS");
                    strcpy(allowable_phase[npha++], "PgS");
                }
                else if (streq(mappedphase, "PnS")) {
                    strcpy(allowable_phase[npha++], "PS");
                    strcpy(allowable_phase[npha++], "PgS");
                }
                else if (streq(mappedphase, "PgS")) {
                    strcpy(allowable_phase[npha++], "PS");
                    strcpy(allowable_phase[npha++], "PnS");
                }
/*
 *              deal with SP et al
 */
                if (streq(mappedphase, "SP")) {
                    strcpy(allowable_phase[npha++], "SPn");
                    strcpy(allowable_phase[npha++], "SPg");
                }
                else if (streq(mappedphase, "SPn")) {
                    strcpy(allowable_phase[npha++], "SP");
                    strcpy(allowable_phase[npha++], "SPg");
                }
                else if (streq(mappedphase, "SPg")) {
                    strcpy(allowable_phase[npha++], "SP");
                    strcpy(allowable_phase[npha++], "SPn");
                }
            }
            if (streq(mappedphase, "PP")) {
                strcpy(allowable_phase[npha++], "PnPn");
                strcpy(allowable_phase[npha++], "PbPb");
                strcpy(allowable_phase[npha++], "PgPg");
            }
            if (streq(mappedphase, "SS")) {
                strcpy(allowable_phase[npha++], "SnSn");
                strcpy(allowable_phase[npha++], "SbSb");
                strcpy(allowable_phase[npha++], "SgSg");
            }
/*
 *
 *          loop through allowable phases and calculate residual
 *
 */
            for (j = 0; j < npha; j++) {
                strcpy(phase, allowable_phase[j]);
/*
 *              only do matching phase types
 */
                if (islower(phase[0])) {
                    if (phase[1] == 'P' || phase[1] == 'w') ttype = 'P';
                    if (phase[1] == 'S') ttype = 'S';
                }
                else
                    ttype = toupper(phase[0]);
                if ((ttype =='P' && stype) || (ttype =='S' && ptype))
                    continue;
/*
 *              do not allow repeating phase names in a reading
 */
                isseen = 0;
                for (ii = rdindx->start; ii < k; ii++)
                    if (streq(p[ii].phase, phase)) isseen = 1;
                if (isseen) continue;
/*
 *              first-arriving P in a reading
 */
                if (p[k].firstP) {
                    if (!isfirstP(phase, mappedphase))
                        continue;
                }
/*
 *              first-arriving S in a reading
 */
                if (p[k].firstS) {
                    if (!isfirstS(phase, mappedphase))
                        continue;
                }
/*
 *              do not allow S(*) phases to be renamed to depth phases (s*)
 */
                if (isS && islower(phase[0]) && phase[1] == 'S') continue;
/*
 *              get travel time for candidate phase
 */
                strcpy(p[k].phase, phase);
                if (read_ttime(sp, &p[k], ec, tt_tables, topo, 0, -1))
                    continue;
/*
 *              keep record of pP ttime
 */
                if (streq(phase, "pP")) pPttime = p[k].ttime;
/*
 *              do not allow pwP if there was no water column correction
 */
                if (streq(phase, "pwP") &&
                    fabs(pPttime - p[k].ttime) < DEPSILON)
                    continue;
/*
 *              time residual
 */
                resid = p[k].time - sp->time - p[k].ttime;
#ifdef SERIAL
                if (verbose > 4)
                    fprintf(logfp, "              %-8s %9.4f %9.4f\n",
                            p[k].phase, p[k].ttime, resid);
#endif
/*
 *              find phase with smallest residual
 */
                if (fabs(resid) < fabs(min_resid)) {
                    strcpy(candidate_phase, p[k].phase);
                    min_resid = resid;
                    ttime = p[k].ttime;
                    dtdd = p[k].dtdd;
                }
            }
/*
 *          if no eligible phase found, set ISC phase code to "".
 */
            if (fabs(min_resid) > bigres) {
                strcpy(p[k].phase, "");
#ifdef SERIAL
                if (verbose > 3) {
                    fprintf(logfp, "            %9d %9d %-6s ",
                            p[k].rdid, p[k].phid, p[k].sta);
                    fprintf(logfp, "cannot identify phase!\n");
                }
#endif
            }

/*
 *          otherwise set to best fitting phase
 */
            else {
#ifdef SERIAL
                if (verbose > 3) {
                    fprintf(logfp, "            %9d %9d %-6s %-8s -> ",
                            p[k].rdid, p[k].phid, p[k].sta, mappedphase);
                    fprintf(logfp, "%-8s %9.4f\n",
                            candidate_phase, min_resid);
                }
#endif
                strcpy(p[k].phase, candidate_phase);
                p[k].resid = min_resid;
                p[k].ttime = ttime;
                p[k].dtdd = dtdd;
            }
        }
    }
}

/*
 *  Title:
 *     isfirstP
 *  Synopsis:
 *     Finds if a phase is in the list of allowable first-arriving P phases.
 *     Uses phase dependent information from <vmodel>_model.txt file.
 *        allowable_first_P - list of phases to which reported first-arriving
 *                         P phases can be renamed
 *        optional_first_P - additional list of phases to which reported
 *                         first-arriving P phases can be renamed
 *  Input Arguments:
 *     phase       - phase
 *     mappedphase - reported phase
 *  Returns 1 if found, 0 otherwise
 *  Called by:
 *     phaseid
 */
static int isfirstP(char *phase, char *mappedphase)
{
    int j;
/*
 *  see if phase is in the list of allowable first-arriving P phases
 */
    for (j = 0; j < firstPphase_num; j++) {
        if (streq(phase, firstPphase[j]))
            return 1;
    }
/*
 *  not in the list of allowable first-arriving P phases;
 *  see if it is in the optional list
 */
    if (j == firstPphase_num && streq(mappedphase, phase)) {
        for (j = 0; j < firstPopt_num; j++) {
            if (streq(phase, firstPopt[j]))
                return 1;
        }
    }
    return 0;
}

/*
 *  Title:
 *     isfirstS
 *  Synopsis:
 *     Finds if a phase is in the list of allowable first-arriving S phases.
 *     Uses phase dependent information from <vmodel>_model.txt file.
 *        allowable_first_S - list of phases to which reported first-arriving
 *                         S phases can be renamed
 *        optional_first_S - additional list of phases to which reported
 *                         first-arriving S phases can be renamed
 *  Input Arguments:
 *     phase       - phase
 *     mappedphase - reported phase
 *  Returns 1 if found, 0 otherwise
 *  Called by:
 *     phaseid
 */
static int isfirstS(char *phase, char *mappedphase)
{
    int j;
/*
 *  see if phase is in the list of allowable first-arriving S phases
 */
    for (j = 0; j < firstSphase_num; j++) {
        if (streq(phase, firstSphase[j]))
            return 1;
    }
/*
 *  not in the list of allowable first-arriving S phases;
 *  see if it is in the optional list
 */
    if (j == firstSphase_num && streq(mappedphase, phase)) {
        for (j = 0; j < firstSopt_num; j++) {
            if (streq(phase, firstSopt[j]))
                return 1;
        }
    }
    return 0;
}

/*
 *  Title:
 *     setsigma
 *  Synopsis:
 *     Sets timedef flag and a priori estimate of measurement error for a phase.
 *     Uses phase dependent information from <vmodel>_model.txt file.
 *        phase_weight - list of a priori measurement errors within specified
 *                delta ranges for IASPEI phases
 *     Sets the time defining flag to true if a valid entry is found in the
 *        phase_weight table.
 *     Makes the phase non-defining if its residual is larger than a threshold,
 *        or it was explicitly set to non-defining by an analyst.
 *     Only phases with timedef = 1 are used in the location.
 *  Input Arguments:
 *     pp    - pointer to phase structure
 *  Output Arguments:
 *     pp    - pointer to phase structure
 *  Called by:
 *     id_pha, reidentify_pha
 */
static void setsigma(PHAREC *pp)
{
    int j;
    double threshold;
/*
 *  set timedef flag and get measurement errors
 */
    pp->timedef = 0;
    pp->measerr = NULLVAL;
    if (pp->time == NULLVAL)
        return;
/*
 *  use the phase_weight structure to get measurement error
 */
    for (j = 0; j < phase_weight_num; j++) {
        if (streq(pp->phase, phase_weight[j].phase)) {
            if (pp->delta >= phase_weight[j].delta1 &&
                pp->delta <  phase_weight[j].delta2) {
                pp->measerr = phase_weight[j].measurement_error;
                if (pp->dupsigma > 0.) pp->measerr += pp->dupsigma;
                pp->timedef = 1;
/*
 *              make phase non-defining if its residual is large
 */
                threshold = sigmathres * pp->measerr;
                if (fabs(pp->resid) > threshold)
                   pp->timedef = 0;
/*
 *              make phase non-defining if editors made it non-defining
 */
                if (pp->force_undef)
                   pp->timedef = 0;
                break;
            }
        }
    }
}

/*
 *  Title:
 *     mark_duplicates
 *  Synopsis:
 *     Identifies and fixes duplicate arrivals.
 *     Once the phases are identified in each reading, it checks for duplicates
 *        reported by various agencies at the same site.
 *     To account for alternative station codes it uses the primary station
 *        code.
 *     Considers time-defining phases only.
 *     Arrival picks are considered duplicate if they are reported at the
 *        same site for the same event and if they arrival time is within
 *        0.1 seconds. For duplicates the arrival time is taken as the mean
 *        of the arrival time, and the phase id is forced to be the same.
 *     If accounting for correlated errors is turned off, duplicates are
 *        explicitly down-weighted. If correlated errors are accounted for,
 *        downweighting is not necessary as duplicates are simply projected
 *        to the null space.
 *     Collects indices of time-defining phases at a site and calls same_sta.
 *  Input Arguments:
 *     sp        - pointer to current solution
 *     p         - array of phase structures
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *  Output Arguments:
 *     p         - array of phase structures
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     eventloc
 *  Calls:
 *     same_sta, print_pha
 */
int mark_duplicates(SOLREC *sp, PHAREC p[], EC_COEF *ec, TT_TABLE *tt_tables,
                    short int **topo)
{
    int samesta[PHA_PER_READ];
    int i, j, k;
/*
 *  create an array samesta of defining phase indexes for each site (prista)
 */
    for (i = 0; i < sp->numphas; i += j) {
        j = 0;        /* Number of phases for this sta. */
/*
 *      deal only with time-defining phases
 */
        if (!p[i].timedef) {
            i++;
            continue;
        }
        for (k = i; k < sp->numphas; k++) {
/*
 *          recall that phases are ordered by delta, prista, rdid, time
 */
            if (strcmp(p[k].prista, p[i].prista))
                break;
            if (!p[k].timedef) {
                i++;
                continue;
            }
            samesta[j++] = k;
/*
 *          check not going past end of samesta array
 */
            if (j > PHA_PER_READ) {
                fprintf(errfp, "mark_duplicates: %s (%s): too many phases\n",
                        p[i].sta, p[i].prista);
                fprintf(logfp, "mark_duplicates: %s (%s): too many phases\n",
                        p[i].sta, p[i].prista);
                return 1;
            }
        }
/*
 *      look for duplicates
 */
        if (j > 1)
            same_sta(samesta, j, sp, p, ec, tt_tables, topo);
    }
    return 0;
}

/*
 *  Title:
 *     same_sta
 *  Synopsis:
 *     Collects indices of time-defining phases at a site arriving at the same
 *        time (within 0.1s tolerance) and calls same_time.
 *     Downweights duplicates if accounting for correlated errors is turned off.
 *  Input Arguments:
 *     samesta   - array of defining phase indices for a single site
 *     n         - size of samesta array
 *     sp        - pointer to current solution
 *     p         - array of phase structures
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *  Output Arguments:
 *     p         - array of phase structures
 *  Called by:
 *     mark_duplicates
 *  Calls:
 *     same_time
 */
static void same_sta(int samesta[], int n, SOLREC *sp, PHAREC p[],
                     EC_COEF *ec, TT_TABLE *tt_tables, short int **topo)
{
    int sametime[PHA_PER_READ];
    int samepha[PHA_PER_READ];
    int done[PHA_PER_READ];
    int i, j, k;
    if (verbose > 3)
        fprintf(logfp, "        same_sta: %s %d phases\n",
                p[samesta[0]].sta, n);
    for (i = 0; i < n; i++) done[i] = 0;
/*
 *  get indices of phases with the same arrival times
 */
    for (i = 0; i < n; i++) {
        if (done[i])
            continue;
        j = 0;                           /* number of phases with this time */
        for (k = i; k < n; k++) {
            if (fabs(p[samesta[k]].time - p[samesta[i]].time) < SAMETIME_TOL) {
                sametime[j++] = samesta[k];
                done[k] = 1;
            }
        }
        if (j > 1)
/*
 *          deal with the duplicates
 */
            same_time(sametime, j, sp, p, ec, tt_tables, topo);
    }
/*
 *  if correlated errors are to be accounted for, we are done here
 */
    if (do_correlated_errors)
       return;

    for (i = 0; i < n; i++) done[i] = 0;
/*
 *  get duplicates with the same phase code
 */
    for (i = 0; i < n; i++) {
        if (done[i])
            continue;
        j = 0;                    /* number of phases with this phase code. */
        for (k = i; k < n; k++) {
            if (streq(p[samesta[k]].phase, p[samesta[i]].phase)) {
                samepha[j++] = samesta[k];
                done[k] = 1;
            }
        }
/*
 *      downweight duplicates
 */
        for (k = 0; k < j; k++) {
            p[samepha[k]].dupsigma = p[samepha[k]].measerr - 1. / (double)j;
            if (verbose > 2)
                fprintf(logfp, "        duplicates: %d %-6s %-8s\n",
                        samepha[k], p[samepha[k]].sta, p[samepha[k]].phase);
        }
    }
}

/*
 *  Title:
 *     same_time
 *  Synopsis:
 *     Sets the arrival time of duplicates to the mean reported arrival time,
 *        and if there are more than one phase names, sets the phase id of all
 *        duplicates to the one with the smallest residual.
 *  Input Arguments:
 *     sametime  - array of phase indexes of duplicate arrivals at a site
 *     n         - size of sametime array
 *     sp        - pointer to current solution
 *     p         - array of phase structures
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *  Output Arguments:
 *     p         - array of phase structures
 *  Called by:
 *     same_sta
 *  Calls:
 *     read_ttime
 */
static void same_time(int sametime[], int n, SOLREC *sp, PHAREC p[],
                      EC_COEF *ec, TT_TABLE *tt_tables, short int **topo)
{
    int match, number_of_codes, min_resid_index, i, j;
    int phacode[PHA_PER_READ];
    double meantime, resid,  min_resid;
    char temp_phase[PHALEN];
    if (verbose > 3)
        fprintf(logfp, "        same_time: %s %f %d phases\n",
                p[sametime[0]].sta, p[sametime[0]].time, n);
/*
 *  get all the different phase codes given for this time
 */
    phacode[0] = sametime[0];
    number_of_codes = 1;
    meantime = p[sametime[0]].time;
    for (i = 1; i < n; i++) {
        meantime += p[sametime[i]].time;
        match = 0;
        for (j = 0; j < number_of_codes; j++) {
            if (streq(p[sametime[i]].phase, p[phacode[j]].phase)) {
                match = 1;
                break;
            }
        }
        if (!match)
            phacode[number_of_codes++] = sametime[i];
    }
    meantime /= (double)n;
    if (verbose > 3)
        fprintf (logfp, "        %d different phases\n", number_of_codes);
/*
 *  set arrival time to mean of reported arrival times
 */
    for (i = 0; i < n; i++) {
        p[sametime[i]].time = meantime;
        if (i) p[sametime[i]].duplicate = 1;
    }
/*
 *  if all the duplicates have the same phase code, we're done
 */
    if (number_of_codes == 1)
        return;
/*
 *  if not, loop over phase codes to find smallest residual
 */
    min_resid = fabs(NULLVAL);
    min_resid_index = phacode[0];
    for (i = 0; i < number_of_codes; i++) {
/*
 *      if it is an initial phase in a reading, stick to it
 */
        if (p[phacode[i]].init) {
            min_resid_index = phacode[i];
            break;
        }
        if (read_ttime(sp, &p[phacode[i]], ec, tt_tables, topo, 0, -1))
            continue;
        resid = fabs(p[phacode[i]].time - sp->time - p[phacode[i]].ttime);
        if (resid < min_resid) {
            min_resid = resid;
            min_resid_index = phacode[i];
        }
        if (verbose > 3)
            fprintf(logfp, "            %6s %f %f\n",
                    p[phacode[i]].phase, p[phacode[i]].ttime, resid);
    }
    if (verbose > 3)
        fprintf(logfp, "            same_time: set to %s\n",
                p[min_resid_index].phase);
/*
 *  set phase codes to the one with the smallest residual
 */
    strcpy(temp_phase, p[min_resid_index].phase);
    for (i = 0; i < n; i++)
        strcpy(p[sametime[i]].phase, temp_phase);
    return;
}


/*
 *  Title:
 *     id_pfake
 *  Synopsis:
 *     PFAKE was used by the NEIC to associate a fake phase name to amplitude
 *     readings. PFAKE is not used in locations, but residuals are useful for
 *     the editors to decide whether the reading is correctly associated to
 *     an event.
 *     Uses phase dependent information from <vmodel>_model.txt file.
 *        allowable_first_P - list of phases to which reported first-arriving
 *                         P phases can be renamed
 *  Input Arguments:
 *     sp        - pointer to current solution
 *     p         - array of phase structures
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *  Output Arguments:
 *     p         - array of phase structures
 *  Called by:
 *     eventloc, fixedhypo
 *  Calls:
 *     read_ttime
 */
void id_pfake(SOLREC *sp, PHAREC p[], EC_COEF *ec, TT_TABLE *tt_tables,
              short int **topo)
{
    int i, j;
    double resid, bigres = 100., min_resid, ttime = NULLVAL;
    char candidate_phase[PHALEN];
/*
 *  loop over all phases for this event.
 */
    for (i = 0; i < sp->numphas; i++) {
/*
 *      only interested in PFAKE phases here
 */
        if (streq(p[i].rep_phase, "PFAKE")) {
/*
 *          get residual w.r.t. first-arriving P
 */
            min_resid = bigres + 1.;
            for (j = 0; j < firstPphase_num; j++) {
                strcpy(p[i].phase, firstPphase[j]);
                if (read_ttime(sp, &p[i], ec, tt_tables, topo, 0, -1))
                    continue;
/*
 *              residual
 */
                resid = p[i].time - sp->time - p[i].ttime;
/*
 *              find phase with smallest residual
 */
                if (fabs(resid) < fabs(min_resid)) {
                    strcpy(candidate_phase, p[i].phase);
                    min_resid = resid;
                    ttime = p[i].ttime;
                }
            }
            if (fabs(min_resid) > bigres) {
/*
 *              if no eligible phase found, set ISC phase code to "".
 */
                strcpy(p[i].phase, "");
                p[i].ttime = NULLVAL;
                p[i].resid = NULLVAL;
            }
            else {
/*
 *              otherwise set to best fitting phase
 */
                strcpy(p[i].phase, candidate_phase);
                p[i].resid = min_resid;
                p[i].ttime = ttime;
            }
        }
    }
}

/*
 *  Title:
 *     remove_pfake
 *  Synopsis:
 *     Remove ISC phase code for temporarily renamed PFAKE phases.
 *  Input Arguments:
 *     sp - pointer to current solution
 *     p  - array of phase structures
 *  Output Arguments:
 *     p         - array of phase structures
 *  Called by:
 *     eventloc, fixedhypo
 */
void remove_pfake(SOLREC *sp, PHAREC p[])
{
    int i;
    for (i = 0; i < sp->numphas; i++) {
        if (streq(p[i].rep_phase, "PFAKE"))
            strcpy(p[i].phase, "");
    }
}

/*
 *  Title:
 *     reported_phase_resid
 *  Synopsis:
 *     Tries to get a residual for a phase that was set to null due to a
 *     large residual. Uses the reported phase name this time.
 *     Uses phase dependent information from <vmodel>_model.txt file.
 *        phase_map - list of possible reported phases with their
 *                    corresponding IASPEI phase id.
 *     Only used once the locator is finished. It is intended to give a
 *     hint to the analyst about the nature of the outlier.
 *  Input Arguments:
 *     sp        - pointer to current solution
 *     pp        - pointer to a phase record
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *  Output Arguments:
 *     pp        - pointer to a phase record
 *  Called by:
 *     calc_resid
 *  Calls:
 *     read_ttime
 */
void reported_phase_resid(SOLREC *sp, PHAREC *pp, EC_COEF *ec,
                          TT_TABLE *tt_tables, short int **topo)
{
    int j;
/*
 *  try reported phase id
 */
    if (pp->rep_phase[0]) {
/*
 *      map reported phase name to IASPEI standard
 */
        for (j = 0; j < phase_map_num; j++) {
            if (streq(pp->rep_phase, phase_map[j].rep_phase)) {
                strcpy(pp->phase, phase_map[j].phase);
                break;
            }
        }
        strcpy(pp->phase, pp->rep_phase);
/*
 *      try to get a residual
 */
        if (read_ttime(sp, pp, ec, tt_tables, topo, 0, -1)) {
            strcpy(pp->phase, "");
            pp->ttime = pp->resid = NULLVAL;
        }
        else {
            pp->resid = pp->time - sp->time - pp->ttime;
        }
    }
    else {
        strcpy(pp->phase, "");
        pp->ttime = pp->resid = NULLVAL;
    }
}

/*  EOF  */
