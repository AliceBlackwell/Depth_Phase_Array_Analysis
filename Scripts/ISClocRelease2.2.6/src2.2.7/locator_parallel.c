#include "iscloc.h"

extern int verbose;
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;
#ifdef WITH_DB
extern PGconn *conn;
#endif
extern struct timeval t0;
extern int max_iter;
extern int min_iter;
extern int min_phases;                               /* min number of phases */
extern double default_depth;        /* used if seed hypocentre depth is NULL */
extern double moho;
extern double conrad;
extern int do_correlated_errors;
extern int allow_damping;
extern int update_db;                            /* write result to database */
extern char isf_outfile[FILENAMELEN];                 /* output ISF filename */
extern double max_depth_km;                          /* max hypocenter depth */
extern double maxdeperror_shallow;
extern double maxdeperror_deep;
extern int do_gridsearch;        /* perform grid search for initial location */
extern int write_gridsearch_results;                /* write results to file */
extern double na_radius;   /* search radius (degrees) around prime epicenter */
extern double na_deptol;            /* search radius (km) around prime depth */
extern double na_ottol;        /* search radius (s) around prime origin time */
extern double na_lpnorm;               /* p-value for norm to compute misfit */
extern int na_itermax;                           /* max number of iterations */
extern int na_nsamplei;                            /* size of initial sample */
extern int na_nsample;                         /* size of subsequent samples */
extern int na_ncells;                     /* number of cells to be resampled */
extern long iseed;                                     /* random number seed */
extern char in_agency[VALLEN];                    /* author for input assocs */

/*
 * Functions:
 *    eventloc
 *    synthetic
 *    fixedhypo
 *    locate_event
 *    getphases
 *    freephaselist
 *    readings
 */

/*
 * Local functions
 *    getndef
 *    getresids
 *    build_gd
 *    project_gd
 *    weight_gd
 *    convtestval
 *    convergence_test
 *    WxG
 */
static int getndef(int numphas, PHAREC p[], int nsta, STAREC stalist[],
                   double *toffset);
static int getresids(SOLREC *sp, READING *rdindx, PHAREC p[], EC_COEF *ec,
                     TT_TABLE *tt_tables, short int **topo, int iszderiv,
                     int *has_depdpres, int *ndef, int *ischanged, int iter,
                     int ispchange, int prevndef, int *nunp, char **phundef,
                     double **dcov, double **w);
static double build_gd(int ndef, SOLREC *sp, PHAREC p[], int fixdepthfornow,
                       double **g, double *d);
static int project_gd(int ndef, int m, double **g, double *d, double **w,
                      double *dnorm, double *wrms);
static int WxG(int j, int ndef, double **w, double **g);
static void weight_gd(int ndef, int m, int numphas, PHAREC p[],
                      double **g, double *d, double *dnorm, double *wrms);
static double convtestval(double gtdnorm, double gnorm, double dnorm);
static int convergence_test(int iter, int m, int *nds, double *sol,
                            double *oldsol, double wrms, double *modelnorm,
                            double *convgtest, double *oldcvgtst,
                            double *step, int *isdiv);

/*
 *  Title:
 *     eventloc
 *  Synopsis:
 *     Prepares for iterative linearised least-squares inversion.
 *        sets starting hypocentre
 *           if hypocenter is fixed then just calculate residuals and return
 *        calculates station separations and nearest-neighbour station order
 *        sets locator option according to instructions:
 *           option = 0 free depth
 *           option = 1 fix to region-dependent default depth
 *           option = 2 fix depth to value provided by analyst (only on request)
 *           option = 3 fix depth to median of reported depths (only on request)
 *           option = 4 fix location (only on request).
 *           option = 5 fix depth and location (only on request).
 *           option = 6 fix hypocentre (only on request).
 *        performs phase identification w.r.t. starting hypocentre
 *        performs NA grid search to get initial guess for linearised inversion
 *        reidentifies phases according to best NA hypocentre
 *        tests for depth resolution; fixes depth if necessary
 *        locates event
 *        if convergent solution is obtained:
 *           discards free-depth solution if depth error is too large
 *              fixes depth and start over again
 *           performs depth-phase stack if there is depth-phase depth resolution
 *           calculates residuals for all reported phases
 *           calculates location quality metrics
 *           calculates magnitudes
 *           writes results to database and/or ISF file
 *        else:
 *           reverts to previous prime hypocentre
 *           calculates residuals for all reported phases
 *        reports results
 *  Input arguments:
 *     isf       - ISF text file input?
 *     database  - database connection exists?
 *     e         - pointer to event info
 *     h         - array of hypocentres
 *     s         - pointer to current solution
 *     p         - array of phase structures
 *     stamag_mb - array of mb stamag structures
 *     stamag_ms - array of Ms stamag structures
 *     rdmag_mb  - array of reading mb structures
 *     rdmag_ms  - array of reading MS structures
 *     mszh      - array of MS vertical/horizontal magnitude structures
 *     ismbQ     - use mb magnitude attenuation table? (0/1)
 *     mbQ       - pointer to mb Q(d,h) table
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     variogram - pointer to generic variogram model
 *     gres      - grid spacing in default depth grid
 *     ngrid     - number of grid points in default depth grid
 *     depthgrid - pointer to default depth grid (lat, lon, depth)
 *     fe        - pointer to FE structure
 *     topo      - ETOPO bathymetry/elevation matrix
 *     isfout    - file pointer to ISF output file
 *     magbloc   - reported magnitudes from ISF input file
 *  Output arguments:
 *     e         - pointer to event info
 *     s         - pointer to current solution
 *     p         - array of phase structures
 *     total     - number of successful locations
 *     fail      - number of failed locations
 *     opt       - locator option counter
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     main
 *  Calls:
 *     gettimeofday, init_event, print_hyp, synthetic, start_hyp,
 *     human_time, get_stalist, distance_matrix, HierarchicalCluster,
 *     init_sol, get_default_depth, calc_delaz, id_pha, mark_duplicates,
 *     calc_resid, depth_phase_check, depth_resolution, depth_phase_stack,
 *     set_searchspace, na_search, reidentify_pha, locate_event, free_matrix,
 *     id_pfake, remove_pfake, location_quality, calc_netmag, gregnum,
 *     print_pha, print_sol, write_isf, put_data, fixedhypo, remove_isc,
 *     replace_prime, replace_assoc
 */
int eventloc(int isf, int database, int *total, int *fail, int *opt,
            EVREC *e, HYPREC h[], SOLREC *s, PHAREC p[],
            STAMAG *stamag_mb, STAMAG *stamag_ms,
            RDMAG *rdmag_mb, RDMAG *rdmag_ms, MSZH *mszh, int ismbQ, MAGQ *mbQ,
            EC_COEF *ec, TT_TABLE *tt_tables, VARIOGRAM *variogram,
            double gres, int ngrid, double **depthgrid,
            FE *fe, double *grn_depth, short int **topo, FILE *isfout,
            char * magbloc)
{
    HYPREC starthyp;                                   /* median hypocenter */
    SOLREC grds;                                         /* solution record */
    HYPQUAL hq;                        /* hypocenter quality metrics record */
    NASPACE nasp;                   /* Neighbourhood algorithm search space */
    double **distmatrix = (double **)NULL;               /* distance matrix */
    STAREC *stalist = (STAREC *)NULL;                /* unique station list */
    STAORDER *staorder = (STAORDER *)NULL;              /* NN station order */
    READING *rdindx = (READING *)NULL;                          /* readings */
#ifdef WITH_DB
    PGresult *res_set = (PGresult *)NULL;
#endif
    HYPREC temp;
    char timestr[25];
    char filename[FILENAMELEN];
    int iszderiv = 0, firstpass = 1, do_gridsearch_cf = do_gridsearch;
    int option = 0, grn = 0, i, iflag;
    int has_depth_resolution = 0, has_depdpres = 0, isdefdep = 0;
    int nsta = 0, ndef = 0;
    double mediandepth, medianot, medianlat, medianlon;
    iflag = do_gridsearch;
/*
 *
 *  Initialize event
 *
 */
    gettimeofday(&t0, NULL);
    if (init_event(e, h)) {
        fprintf(logfp, "ABORT: event initialization failed!\n");
        return 1;
    }
/*
 *  Print hypocentres (first is prime, the rest is ordered by score)
 */
    print_hyp(e, h);
/*
 *  Check that there are enough phases to start with.
 */
    if (!e->hypo_fix && e->numphas < min_phases) {
        fprintf(logfp, "WARNING: insufficient number of phases (%d), ",
                e->numphas);
        fprintf(logfp, "fixing the hypocentre\n");
        fprintf(errfp, "insufficient number of phases (%d), fixing the hypocentre",
                e->numphas);
        e->hypo_fix = 1;
    }
/*
 *  Allocate memory for rdindx
 */
    if ((rdindx = (READING *)calloc(e->numrd, sizeof(READING))) == NULL) {
        fprintf(logfp, "ABORT: cannot allocate memory for rdindx!\n");
        fprintf(errfp, "eventloc: cannot allocate memory for rdindx!\n");
        errorcode = 1;
        return 1;
    }
    readings(e->numphas, e->numrd, p, rdindx);
/*
 *
 *  If hypocenter is fixed then just calculate residuals
 *
 */
    if (e->hypo_fix) {
        fprintf(logfp, "Calculate residuals for fixed hypocentre\n");
        synthetic(e, &h[0], s, rdindx, p, ec, tt_tables, topo, database, isf);
        opt[6]++;
        Free(rdindx);
        return 0;
    }
/*
 *  set starting hypocentre to median of all hypocentre parameters
 */
    start_hyp(e, h, &starthyp);
    human_time(timestr, starthyp.time);
    fprintf(logfp, "Median hypocentre:\n");
    fprintf(logfp, "  OT = %s Lat = %7.3f Lon = %8.3f Depth = %.1f\n",
            timestr, starthyp.lat, starthyp.lon, starthyp.depth);
/*
 *  get unique station list
 */
    if ((stalist = get_stalist(e->numphas, p, &nsta)) == NULL) {
        fprintf(logfp, "ABORT: cannot generate station list!\n");
        fprintf(errfp, "cannot get station list!\n");
        Free(rdindx);
        return 1;
    }
/*
 *  correlated errors
 */
    if (do_correlated_errors) {
        if ((staorder = (STAORDER *)calloc(nsta, sizeof(STAORDER))) == NULL) {
            fprintf(logfp, "ABORT: staorder: cannot allocate memory\n");
            fprintf(errfp, "staorder: cannot allocate memory\n");
            Free(stalist);
            Free(rdindx);
            return 1;
        }
/*
 *      calculate station separations
 */
        if ((distmatrix = distance_matrix(nsta, stalist)) == NULL) {
            fprintf(logfp, "ABORT: distance_matrix failed!\n");
            fprintf(errfp, "cannot get distmatrix!\n");
            Free(stalist); Free(staorder);
            Free(rdindx);
            return 1;
        }
/*
 *      nearest-neighbour station order
 */
        if (HierarchicalCluster(nsta, distmatrix, staorder)) {
            fprintf(logfp, "ABORT: HierarchicalCluster failed!\n");
            fprintf(errfp, "HierarchicalCluster failed!\n");
            Free(stalist); Free(staorder);
            free_matrix(distmatrix);
            Free(rdindx);
            return 1;
        }
    }
/*
 *  Option loop: set options
 *       option = 0 free depth
 *       option = 1 fix to region-dependent default depth
 *       option = 2 fixed depth by analyst (only on request)
 *       option = 3 fix depth to median of reported depths (only on request)
 *       option = 4 fix location (only on request).
 *       option = 5 fix depth and location (only on request).
 */
    for (option = 0; option < 2; option++) {
        errorcode = 0;
/*
 *      Fixed depth instruction
 */
        if (e->depth_fix)
            option = 2;
/*
 *      Default depth instruction
 */
        if (e->fix_depth_default)
            option = 1;
/*
 *      Fix on median depth instruction
 */
        if (e->fix_depth_median)
            option = 3;
/*
 *      Fixed location instruction
 */
        if (e->epi_fix) {
/*
 *          depending on e.depth_fix solve for OT only or OT and depth
 */
            if (e->depth_fix) option = 5;
            else              option = 4;
        }
        fprintf(logfp, "Option %d\n", option);
/*
 *      Number of model parameters
 */
        if      (option == 0) s->number_of_unknowns = 4;
        else if (option == 4) s->number_of_unknowns = 2;
        else if (option == 5) s->number_of_unknowns = 1;
        else                  s->number_of_unknowns = 3;
        s->number_of_unknowns -= e->time_fix;
/*
 *      depth derivatives are only needed when depth is a free parameter
 */
        if (option == 0 || option == 4) iszderiv = 1;
        else                            iszderiv = 0;
/*
 *      depth fix type (option 1 is set by get_default_depth)
 */
        if (option == 0 || option == 4)
            s->depfixtype = 0;
        if (option == 2 || option == 5) {
            if (e->depth_fix_editor)
                s->depfixtype = 8;
            else if (e->fix_depth_depdp)
                s->depfixtype = 3;
            else if (e->fix_depth_default)
                s->depfixtype = 5;
            else if (e->fix_depth_median) {
                s->depfixtype = 6;
                isdefdep = 1;
            }
            else if (e->surface_fix)
                s->depfixtype = 4;
            else
                s->depfixtype = 2;
        }
        if (option == 3) {
            s->depfixtype = 6;
            isdefdep = 1;
        }
/*
 *      initialize the solution structure
 */
        if (firstpass) {
            if (init_sol(s, e, &starthyp)) {
                fprintf(logfp, "    WARNING: init_sol failed!\n");
                continue;
            }
            mediandepth = s->depth;
            medianot = s->time;
            medianlat = s->lat;
            medianlon = s->lon;
        }
        if (option == 1) {
/*
 *          either no depth resolution or the locator did not converge
 *              fix depth to region-dependent default depth
 */
            s->depth = mediandepth;
            s->depth = get_default_depth(s, ngrid, gres, depthgrid,
                                         fe, grn_depth, &isdefdep);
/*
 *          if there is a large depth difference, fall back to the
 *              initial hypocentre and recalcaluate the default depth
 */
            if (fabs((s->depth - mediandepth)) > 20.) {
                fprintf(logfp, "Large depth difference, ");
                fprintf(logfp, "fall back to median hypocentre\n");
                s->time = medianot;
                s->lat = medianlat;
                s->lon = medianlon;
                s->depth = mediandepth;
                s->depth = get_default_depth(s, ngrid, gres, depthgrid,
                                             fe, grn_depth, &isdefdep);
            }
/*
 *          adjust origin time to depth change
 */
            s->time += (s->depth - mediandepth) / 10.;
/*
 *          if there is still a large depth difference, do NA again
 */
            if (fabs((s->depth - mediandepth)) > 20.)
                firstpass = 1;
        }
        if (option == 0 || option == 4) s->depfix = 0;
        else                            s->depfix = 1;
        human_time(timestr, s->time);
        fprintf(logfp, "Initial hypocentre:\n");
        fprintf(logfp, "  OT = %s Lat = %7.3f Lon = %8.3f Depth = %.1f\n",
                timestr, s->lat, s->lon, s->depth);
/*
 *      delta, esaz and seaz for each phase
 */
        calc_delaz(s, p, 1);
/*
 *      ISC phase identification
 */
        ndef = id_pha(s, rdindx, p, ec, tt_tables, topo);
/*
 *      deal with duplicate readings
 */
        if (mark_duplicates(s, p, ec, tt_tables, topo))
            continue;
/*
 *
 *      Neighbourhood algorithm search to get initial hypocentre guess
 *          - may be executed only once
 *          - search in 4D (lat, lon, OT, depth)
 *          - reidentify phases w.r.t. each trial hypocentre
 *          - ignore correlated error structure for the sake of speed
 */
        if (do_gridsearch && firstpass) {
            fprintf(logfp, "Neighbourhood algorithm (%.4f)\n", secs(&t0));
            memmove(&grds, s, sizeof(SOLREC));
/*
 *          set up search space for NA
 */
            if (set_searchspace(&grds, &nasp)) {
                fprintf(logfp, "    WARNING: set_searchspace failed!\n");
            }
            else {
/*
 *              Neighbourhood algorithm
 */
                if (write_gridsearch_results)
                    sprintf(filename, "%d.%d.gsres", e->evid, option);
                if (na_search(nsta, &grds, p, tt_tables, ec, topo, stalist,
                            distmatrix, variogram, staorder, &nasp, filename)) {
                    fprintf(logfp, "    WARNING: na_search failed!\n");
                    memmove(&grds, s, sizeof(SOLREC));
                }
                else {
/*
 *                  store the best hypo from NA in the solution record
 */
                    s->lat = grds.lat;
                    s->lon = grds.lon;
                    s->time = grds.time;
                    s->depth = grds.depth;
                    human_time(timestr, s->time);
                    fprintf(logfp, "Best fitting hypocentre from grid search:\n");
                    fprintf(logfp, "  OT = %s Lat = %7.3f Lon = %8.3f ",
                            timestr, s->lat, s->lon);
                    fprintf(logfp, "Depth = %.1f\n", s->depth);
/*
 *                  reidentify phases w.r.t. best hypocentre
 */
                    fprintf(logfp, "Reidentify phases after NA\n");
                    calc_delaz(s, p, 0);
                    reidentify_pha(s, rdindx, p, ec, tt_tables, topo);
                    mark_duplicates(s, p, ec, tt_tables, topo);
                }
                fprintf(logfp, "NA (%.4f) done\n", secs(&t0));
            }
        }
/*
 *      disable further grid searches
 */
        firstpass = 0;
/*
 *      set ttime, residual, dtdh, and dtdd for each phase
 */
        if (calc_resid(s, p, "use", ec, tt_tables, topo, iszderiv))
            continue;
/*
 *      number of initial time-defining phases
 */
        ndef = 0;
        for (i = 0; i < s->numphas; i++)
            if (p[i].timedef) ndef++;
        if (ndef < s->number_of_unknowns) {
            fprintf(logfp, "Insufficient number (%d) of phases left\n", ndef);
            errorcode = 7;
            continue;
        }
/*
 *      depth-phase depth resolution
 *          (ndepassoc >= mindepthpha && nagent >= ndepagency)
 *          also flag first arriving defining P for a reading
 */
        i = (option == 0 || option == 4) ? 1 : 0;
        has_depdpres = depth_phase_check(s, rdindx, p, i);
/*
 *      recount number of time-defining phases as depth_phase_check
 *          may make depth phases non-defining
 */
        ndef = 0;
        for (i = 0; i < s->numphas; i++)
            if (p[i].timedef) ndef++;
        if (ndef < s->number_of_unknowns) {
            fprintf(logfp, "Insufficient number (%d) of phases left\n", ndef);
            errorcode = 7;
            continue;
        }
/*
 *      depth resolution
 *          (has_depdpres || nlocal >= minlocalsta ||
 *           nsdef >= min_s_p || ncoredef >= min_corepha)
 */
        i = (option == 0 || option == 4) ? 1 : 0;
        has_depth_resolution = depth_resolution(s, rdindx, p, i);
        if (has_depdpres) has_depth_resolution = 1;
/*
 *      pointless to try free-depth solution without depth resolution
 */
        if (!has_depth_resolution && (option == 0 || option == 4)) {
            fprintf(logfp, "No depth resolution for free-depth solution!\n");
            if (fabs(s->depth - mediandepth) > 20.) firstpass = 1;
            continue;
        }
/*
 *
 *      locate event
 *
 */
        fprintf(logfp, "Event location\n");
        if (locate_event(option, nsta, has_depdpres, s, rdindx, p, ec,
                         tt_tables, stalist, distmatrix, variogram, staorder,
                         topo)) {
/*
 *          divergent solution
 */
            if (iflag) {
/*
 *              if grid search was enabled:
 *                  disable it, reinitialize and give it one more try
 */
                do_gridsearch = 0;
                iflag = 0;
                firstpass = 1;
                option--;
                if (option > 0) option = 0;
                fprintf(logfp, "Try again without the grid search\n");
            }
            else if (option == 0)
/*
 *              reinitialize solution for the fixed depth option
 */
                firstpass = 1;
            else {
/*
 *              give up
 */
                fprintf(logfp, "locator failed!\n");
                fprintf(errfp, "locator failed!\n");
            }
            continue;
        }
/*
 *
 *      converged?
 *
 */
        if (s->converged) {
/*
 *          Discard free-depth solution if depth error is too large
 */
            if (option == 0 && (
                     (s->depth > 0. && s->depth <= moho &&
                      s->error[3] > maxdeperror_shallow) ||
                     (s->depth >  moho && s->error[3] > maxdeperror_deep))) {
                fprintf(logfp, "Discarded free-depth solution!\n");
                fprintf(logfp, "     depth = %5.1f depth error = %.1f\n",
                                s->depth, s->error[3]);
                firstpass = 1;
                continue;
            }
/*
 *          We're done.
 */
            else
                break;
        }
    }
/*
 *  End of option loop
 */
    do_gridsearch = do_gridsearch_cf;
    if (verbose)
        fprintf(logfp, "End of option loop (%.4f)\n", secs(&t0));
/*
 *  Free memory
 */
    Free(stalist);
    if (do_correlated_errors) {
        free_matrix(distmatrix);
        Free(staorder);
    }
/*
 *
 *  Convergence: calculate all residuals, magnitudes etc.
 *
 */
    if (s->converged) {
        fprintf(logfp, "Convergent solution, final touches\n");
/*
 *      Calculate depth-phase depth if possible.
 */
        s->depdp = s->depdp_error = NULLVAL;
        has_depdpres = depth_phase_check(s, rdindx, p, 1);
        if (has_depdpres) {
            depth_phase_stack(s, p, tt_tables, topo);
            if (s->depdp != NULLVAL)
                fprintf(logfp, "    ndp = %d, depth=%.1f, depth error=%.1f\n",
                        s->ndp, s->depdp, s->depdp_error);
        }
/*
 *      Temporarily reidentify PFAKES so that they get residuals.
 */
        id_pfake(s, p, ec, tt_tables, topo);
/*
 *      Calculate residuals for all phases
 */
        calc_resid(s, p, "all", ec, tt_tables, topo, 0);
/*
 *      Remove ISC phase codes from temporarily reidentified PFAKES.
 */
        remove_pfake(s, p);
/*
 *      Calculate location quality metrics
 */
        if (!location_quality(s->numphas, p, &hq)) {
            s->azimgap = hq.whole_net.gap;
            s->mindist = hq.whole_net.mindist;
            s->maxdist = hq.whole_net.maxdist;
        }
/*
 *      Magnitudes
 */
        calc_netmag(s, rdindx, p, stamag_mb, stamag_ms, rdmag_mb, rdmag_ms,
                    mszh, ismbQ, mbQ);
        Free(rdindx);
/*
 *      Print final solution, phases and residuals to log file.
 */
        grn = gregnum(s->lat, s->lon, fe);
        print_pha(s->numphas, p);
        fprintf(logfp, "final : ");
        print_sol(s, grn);
        fprintf(logfp, "    etype=%s nreading=%d nass=%d ndef=%d ndefsta=%d ",
                e->etype, s->nreading, s->nass, s->ndef, s->ndefsta);
        fprintf(logfp, "nrank=%d", s->prank);
        if (s->sdobs != NULLVAL)
            fprintf(logfp, " sdobs=%.3f", s->sdobs);
        fprintf(logfp, " sgap=%5.1f\n", hq.whole_net.sgap);
        if (s->smajax != NULLVAL)
            fprintf(logfp, "    smajax=%.1f sminax=%.1f strike=%.1f",
                            s->smajax, s->sminax, s->strike);
        if (s->error[0] != NULLVAL)
            fprintf(logfp, " stime=%.3f", s->error[0]);
        if (s->error[3] != NULLVAL)
            fprintf(logfp, " sdepth=%.1f", s->error[3]);
        fprintf(logfp, "\n");
        if (s->depdp != NULLVAL)
            fprintf(logfp, "    depdp=%.2f +/- %.2f ndp=%d\n",
                            s->depdp, s->depdp_error, s->ndp);
        if (e->epi_fix)
            fprintf(logfp, "    epicentre fixed to %s epicentre\n",
                    e->location_agency);
        if (e->time_fix)
            fprintf(logfp, "    origin time fixed to %s origin time\n",
                    e->time_agency);
        if (s->depfixtype == 8)
            fprintf(logfp, "    depth fixed by editor\n");
        else if (s->depfixtype == 1)
            fprintf(logfp, "    airquake, depth fixed to surface\n");
        else if (s->depfixtype == 2)
            fprintf(logfp, "    depth fixed to %s depth\n",
                    e->depth_agency);
        else if (s->depfixtype == 3)
            fprintf(logfp, "    depth fixed to depth-phase depth\n");
        else if (s->depfixtype == 4)
            fprintf(logfp, "    anthropogenic event, depth fixed to surface\n");
        else if (s->depfixtype == 5)
            fprintf(logfp, "    depth fixed to default depth grid depth\n");
        else if (s->depfixtype == 6) {
            if (!isdefdep) {
                fprintf(logfp, "    no default depth grid point exists, ");
                fprintf(logfp, "depth fixed to median reported depth\n");
            }
            else {
                fprintf(logfp, "    depth fixed to median reported depth\n");
            }
        }
        else if (s->depfixtype == 7) {
            fprintf(logfp, "    no default depth grid point exists, ");
            fprintf(logfp, "depth fixed to GRN-dependent depth\n");
        }
        else
            fprintf(logfp, "    free-depth solution\n");
        fprintf(logfp, "    local network:  nsta=%d ndef=%d sgap=%5.1f",
                hq.local_net.ndefsta, hq.local_net.ndef, hq.local_net.sgap);
        fprintf(logfp, " dU=%5.3f ndefsta_10km=%d GT5cand=%d\n",
                hq.local_net.du, hq.ndefsta_10km, hq.gtcand);
        if (s->bodymag != NULLVAL)
            fprintf(logfp, "    mb=%.2f +/- %.2f nsta=%d\n",
                            s->bodymag, s->bodymag_uncertainty, s->nsta_mb);
        if (s->surfmag != NULLVAL)
            fprintf(logfp, "    MS=%.2f +/- %.2f nsta=%d\n",
                            s->surfmag, s->surfmag_uncertainty, s->nsta_ms);
/*
 *      Update counters
 */
        opt[option]++;
        *total += 1;
/*
 *      Write solution
 */
#ifdef WITH_DB
/*
 *      Write solution to database if required.
 */
        if (update_db && database) {
            res_set = PQexec(conn, "BEGIN");
            PQclear(res_set);
            if (verbose)
                fprintf(logfp, "        put_data (%.4f)\n", secs(&t0));
            if (put_data(e, s, p, &hq, stamag_mb, stamag_ms,
                        rdmag_mb, rdmag_ms, mszh, fe)) {
                fprintf(errfp, "eventloc: could not update event %d!\n",
                        e->evid);
                fprintf(logfp, "eventloc: could not update event %d!\n",
                        e->evid);
                res_set = PQexec(conn, "ROLLBACK");
                PQclear(res_set);
            }
            else {
                res_set = PQexec(conn, "COMMIT");
                PQclear(res_set);
                fprintf(logfp, "eventloc: updated event %d (%.2f)\n",
                        e->evid, secs(&t0));
            }
        }
        else {
            if (update_db) {
                fprintf(errfp, "eventloc: no database connection!\n");
                fprintf(logfp, "eventloc: no database connection!\n");
            }
        }
#endif
/*
 *      Write event to ISF2 file if required.
 */
        if (isf_outfile[0]) {
            i = s->hypid;
            s->hypid = *total;
            write_isf(isfout, e, s, h, p, stamag_mb, stamag_ms,
                      rdmag_mb, rdmag_ms, grn, magbloc);
            s->hypid = i;
        }
    }
/*
 *
 *  Non-convergence: roll back to previous prime
 *
 */
    else {
        *fail += 1;
/*
 *      no need to do anything if data read from ISF file
 */
        if (isf) {
            Free(rdindx);
            fprintf(logfp, "Finished event %d (%.2f)\n", e->evid, secs(&t0));
            return 1;
        }
/*
 *      Get rid of any starting point from instructions.
 */
        e->start_depth = NULLVAL;
        e->start_lat   = NULLVAL;
        e->start_lon   = NULLVAL;
        e->start_time  = NULLVAL;
/*
 *      Check if ISC solution is the previous prime
 */
        if (streq(h[0].agency, in_agency)) {
            if (e->numhyps > 1) {
/*
 *              revert to a non-ISC prime
 */
                fprintf(logfp, "Set prime to %s\n", h[1].agency);
                swap(h[1], h[0]);
            }
            else {
/*
 *              ISC is the only hypocenter; generate warning
 */
                fprintf(logfp, "WARNING: Search event %d could not ", e->evid);
                fprintf(logfp, "converge and should be banished\n");
                fprintf(logfp, "FAILURE\n");
                Free(rdindx);
                return 1;
            }
        }
/*
 *      Reset solution to previous prime and calculate residuals
 */
        fprintf(logfp, "Calculate residuals w.r.t. previous prime\n");
        fixedhypo(e, &h[0], s, rdindx, p, ec, tt_tables, topo);
        print_pha(s->numphas, p);
        Free(rdindx);
        fprintf(logfp, "FAILURE\n");
/*
 *      update DB
 */
#ifdef WITH_DB
        if (update_db && database) {
            res_set = PQexec(conn, "BEGIN");
            PQclear(res_set);
            if (verbose)
                fprintf(logfp, "    eventloc: prime: %d new: %d evid: %d\n",
                                e->prime, h[0].hypid, e->evid);
/*
 *          Remove ISC hypocentre
 */
            if (e->isc_hypid != NULLVAL) {
                fprintf(logfp, "Removing ISC hypocentre\n");
                remove_isc(e);
            }
/*
 *          change prime if necessary
 */
            if (e->prime != h[0].hypid) {
                if (verbose) fprintf(logfp, "    replace_prime\n");
                replace_prime(e, &h[0]);
            }
            if (verbose) fprintf(logfp, "    replace_assoc\n");
            replace_assoc(e, p, &h[0]);
            res_set = PQexec(conn, "COMMIT");
            PQclear(res_set);
        }
        else {
            if (update_db) {
                fprintf(errfp, "eventloc: no database connection!\n");
                fprintf(logfp, "eventloc: no database connection!\n");
            }
        }
#endif
        return 1;
    }
    fprintf(logfp, "Finished event %d (%.2f)\n", e->evid, secs(&t0));
    return 0;
}

/*
 *  Title:
 *     synthetic
 *  Synopsis:
 *     Calculate residuals w.r.t. a hypocentre.
 *     Sets prime hypocentre in DB if necessary.
 *  Input Arguments:
 *     ep        - pointer to event info
 *     hp        - pointer to hypocentre
 *     sp        - pointer to current solution
 *     rdindx    - array of reading structures
 *     p         - array of phase structures
 *     ec        - ellipticity correction coefs
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *     database  - database connection exists?
 *     isf       - ISF text file input?
 *  Called by:
 *     eventloc
 *  Calls:
 *     fixedhypo, replace_prime, replace_assoc, print_pha
 */
void synthetic(EVREC *ep, HYPREC *hp, SOLREC *sp, READING *rdindx, PHAREC p[],
               EC_COEF *ec, TT_TABLE *tt_tables, short int **topo,
               int database, int isf)
{
/*
 *  Set solution to favourite hypocenter and calculate residuals
 */
    fprintf(logfp, "Calculating residuals for a fixed hypocentre\n");
    sp->number_of_unknowns = 0;
    fixedhypo(ep, hp, sp, rdindx, p, ec, tt_tables, topo);
/*
 *  Print phases and residuals to log file.
 */
    print_pha(sp->numphas, p);
    if (isf) return;
/*
 *  Change association rows in database.
 */
#ifdef WITH_DB
    if (update_db && database) {
/*
 *      change prime if necessary
 */
        if (ep->prime != hp->hypid) {
            if (verbose) fprintf(logfp, "    replace_prime\n");
            replace_prime(ep, hp);
        }
        if (verbose) fprintf(logfp, "    replace_assoc\n");
        replace_assoc(ep, p, hp);
    }
    else {
        if (update_db) {
            fprintf(errfp, "synthetic: no database connection!\n");
            fprintf(logfp, "synthetic: no database connection!\n");
        }
    }
#endif
}

/*
 *  Title:
 *     fixedhypo
 *  Synopsis:
 *     Calculate residuals w.r.t. a fixed hypocentre.
 *  Input Arguments:
 *     ep        - pointer to event info
 *     hp        - pointer to hypocentre
 *     sp        - pointer to current solution
 *     rdindx    - array of reading structures
 *     p[]       - array of phase structures
 *     ec        - ellipticity correction coefs
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *  Called by:
 *     eventloc, synthetic
 *  Calls:
 *     init_sol, calc_delaz, id_pha, id_pfake, calc_resid, remove_pfake
 */
void fixedhypo(EVREC *ep, HYPREC *hp, SOLREC *sp, READING *rdindx, PHAREC p[],
               EC_COEF *ec, TT_TABLE *tt_tables, short int **topo)
{
/*
 *  Set solution to fixed hypo
 */
    if (verbose) fprintf(logfp, "    init_sol\n");
    if (hp->depth == NULLVAL || hp->depth < 0.)
        hp->depth = default_depth;
    init_sol(sp, ep, hp);
/*
 *  Set delta and seaz for each phase.
 */
    if (verbose) fprintf(logfp, "    calc_delaz\n");
    calc_delaz(sp, p, 1);
/*
 *  Identify phases.
 */
    if (verbose) fprintf(logfp, "    id_pha\n");
    id_pha(sp, rdindx, p, ec, tt_tables, topo);
/*
 *  Temporarily reidentify PFAKES so that they get residuals.
 */
    if (verbose) fprintf(logfp, "    id_pfake\n");
    id_pfake(sp, p, ec, tt_tables, topo);
/*
 *  Calculate residuals (no need for dtdd and dtdh)
 */
    if (verbose) fprintf(logfp, "    calc_resid\n");
    calc_resid(sp, p, "all", ec, tt_tables, topo, 0);
/*
 *  Remove ISC phase codes from temporarily reidentified PFAKES.
 */
    if (verbose) fprintf(logfp, "    remove_pfake\n");
    remove_pfake(sp, p);
}

/*
 *  Title:
 *     locate_event
 *  Synopsis:
 *     Iterative linearised least-squares inversion of travel-times to
 *         obtain a solution for the hypocentre.
 *     Bondár, I., and K. McLaughlin, 2009,
 *        Seismic location bias and uncertainty in the presence of correlated
 *        and non-Gaussian travel-time errors,
 *        Bull. Seism. Soc. Am., 99, 172-193.
 *     Bondár, I., and D. Storchak, 2011,
 *        Improved location procedures at the International Seismological
 *        Centre,
 *        Geophys. J. Int., doi: 10.1111/j.1365-246X.2011.05107.x.
 *
 *     If do_correlated_errors is true, it projects Gm = d into the
 *         eigensystem defined by the full data covariance matrix,
 *         i.e. in the system where the full data covariance matrix
 *         becomes diagonal
 *     otherwise assumes independent errors and weights Gm = d with
 *         the a priori estimates of measurement error variances.
 *     WGm = Wd is solved with singular value decomposition.
 *     Damping is applied if condition number is large.
 *     Convergence test is based on the Paige-Saunder convergence test value
 *         and the history of model and data norms.
 *     Formal uncertainties (model covariance matrix) are scaled
 *         to <confidence>% confidence level.
 *     Free-depth solutions:
 *         Depth is fixed in the first min_iter-1 iterations.
 *         Depth remains fixed if number of airquakes/deepquakes exceeds 2;
 *             depthfixtype is set to 'B'.
 *         Phases are reidentified if depth crosses Moho or Conrad
 *             discontinuities.
 *     Correlated errors:
 *         The data covariance and projection matrices are calculated once.
 *         The data covariance matrix is recalculated if defining phases were
 *             renamed during an iteration.
 *         The projection matrix is recalculated if defining phases were
 *            renamed or defining phases were made non-defining during an
 *            iteration.
 *  Input arguments:
 *     option       - locator option
 *                    option 0 = free depth
 *                    option 1 = depth fixed to default regional depth
 *                    option 2 = fixed depth by analyst
 *                    option 3 = depth fixed to median depth
 *                    option 4 = fixed location (lat, lon)
 *                    option 5 = fixed depth and location
 *     nsta         - number of distinct stations
 *     has_depdpres - do we have depth resolution by depth phases?
 *     sp           - pointer to current solution
 *     rdindx       - array of reading structures
 *     p            - array of phase structures
 *     ec           - pointer to ellipticity correction coefficient structure
 *     tt_tables    - pointer to travel-time tables
 *     stalist      - array of starec structures
 *     distmatrix   - station separation matrix
 *     variogramp   - pointer to generic variogram model
 *     staorder     - array of staorder structures (nearest-neighbour order)
 *     gres         - grid spacing in default depth grid
 *     ngrid        - number of grid points in default depth grid
 *     depthgrid    - pointer to default depth grid (lat, lon, depth)
 *     topo         - ETOPO bathymetry/elevation matrix
 *  Output arguments:
 *     sp           - pointer to current solution.
 *     p            - array of phase structures
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     eventloc
 *  Calls:
 *     getndef, sort_phaserec_nn, depth_phase_check, print_pha,
 *     calc_delaz, reidentify_pha, mark_duplicates, getresids,
 *     alloc_matrix, data_covariance_matrix, projection_matrix,
 *     free_matrix, build_gd, project_gd, weight_gd,
 *     svd_decompose, svd_threshold, svd_rank, svd_norm, svd_solve,
 *     convtestval, convergence_test, deltaloc, print_sol, print_defining_pha,
 *     sort_phaserec_db, svd_model_covariance_matrix, calc_error
 */
int locate_event(int option, int nsta, int has_depdpres, SOLREC *sp,
                 READING *rdindx, PHAREC p[], EC_COEF *ec, TT_TABLE *tt_tables,
                 STAREC stalist[], double **distmatrix,
                 VARIOGRAM *variogramp, STAORDER staorder[], short int **topo)
{
    int i, j, m, iter, iserr = 0, nds[3], isconv = 0, isdiv = 0;
    int iszderiv = 0, fixdepthfornow = 0, nairquakes = 0, ndeepquakes = 0;
    int prank = 0, dpok = 0, ndef = 0, nd = 0, nr = 0, nunp = 0;
    int ischanged = 0, ispchange = 0;
    double **g = (double **)NULL;
    double *d = (double *)NULL;
    double *sv = (double *)NULL;
    double svundamped[4];
    double **v = (double **)NULL;
    double **dcov = (double **)NULL;
    double **w = (double **)NULL;
    double mcov[4][4], sol[4], oldsol[4], modelnorm[3], convgtest[3];
    double toffset = 0., torg = 0., delta = 0., azim = 0.;
    double svth = 0., damp = 0., dmax = 0., step = 0., prev_depth = 0.;
    double urms = 0., wrms = 0., scale = 0.;
    double gtd = 0., gtdnorm = 0., dnorm = 0., gnorm = 0., mnorm = 0.;
    double cnvgtst = 0., oldcvgtst = 0., cond = 0., loss = 0.;
    char *phundef[MAXNUMPHA], phbuf[MAXNUMPHA * PHALEN];
    for (i = 0; i < MAXNUMPHA; i++) phundef[i] = phbuf + i * PHALEN;
    dpok = has_depdpres;
    prev_depth = sp->depth;
/*
 *  find number of defining phases and earliest arrival time
 */
    nd = getndef(sp->numphas, p, nsta, stalist, &toffset);
    prank = nd;
    if (nd < 0) return 1;
    if (nd <= sp->number_of_unknowns) {
        if (verbose)
            fprintf(logfp, "locate_event: insufficient number of phases (%d)!\n",
                    nd);
        errorcode = 5;
        return 1;
    }
/*
 *  reduced origin time
 */
    torg = sp->time - toffset;
/*
 *  initializations
 */
    for (i = 0; i < 4; i++) {
        sol[i] = 0.;
        oldsol[i] = 0.;
    }
    i = 0;
    if (!sp->timfix)
        sol[i++] = torg;
    if (!sp->epifix) {
        sol[i++] = sp->lon;
        sol[i++] = sp->lat;
    }
    if (!sp->depfix)
        sol[i] = sp->depth;
    nairquakes = ndeepquakes = 0;
    ischanged = ispchange = isconv = isdiv = 0;
    iserr = 1;
    step = oldcvgtst = 1.;
    for (i = 0; i < 4; i++) {
        sp->error[i] = sp->covar[i][i] = NULLVAL;
        for (j = i + 1; j < 4; j++)
            sp->covar[i][j] = sp->covar[j][i] = NULLVAL;
    }
/*
 *  convergence history
 */
    for (j = 0; j < 3; j++) {
        modelnorm[j] = 0.;
        convgtest[j] = 0.;
        nds[j] = 0;
    }
    nds[0] = nd;
/*
 *  reorder phaserecs by staorder, rdid, time so that covariance matrices
 *  for various phases will become block-diagonal
 */
    if (do_correlated_errors) {
        sort_phaserec_nn(sp->numphas, nsta, p, stalist, staorder);
        readings(sp->numphas, sp->nreading, p, rdindx);
        dpok = depth_phase_check(sp, rdindx, p, 0);
        if (verbose > 3)
            print_pha(sp->numphas, p);
    }
/*
 *
 * iteration loop
 *
 */
    if (verbose)
        fprintf(logfp, "Start iteration loop (%.4f)\n", secs(&t0));
    for (iter = 0; iter < max_iter; iter++) {
        if (verbose) fprintf(logfp, "iteration = %d\n", iter);
/*
 *      number of model parameters
 */
        m = sp->number_of_unknowns;
        iszderiv = 1;
/*
 *      check if necessary to fix depth if free depth solution
 */
        if (option == 0 || option == 4) {
            fixdepthfornow = 0;
/*
 *          fix depth for the first min_iter - 1 iterations
 */
            if (iter < min_iter - 1) {
                fixdepthfornow = 1;
                m--;
            }
/*
 *          do not allow airquakes
 */
            else if (sp->depth < 0.) {
                if (verbose)
                    fprintf(logfp, "    airquake, fixing depth to 0\n");
                nairquakes++;
                sp->depth = 0.;
                fixdepthfornow = 1;
                m--;
            }
/*
 *          do not allow deepquakes
 */
            else if (sp->depth > max_depth_km) {
                if (verbose)
                    fprintf(logfp, "    deepquake, fixing depth to max depth\n");
                ndeepquakes++;
                sp->depth = max_depth_km;
                fixdepthfornow = 1;
                m--;
            }
        }
/*
 *      various fix depth instructions
 */
        else
            fixdepthfornow = 1;
/*
 *      enough of airquakes!
 */
        if (nairquakes > 2 || ndeepquakes > 2) {
            fixdepthfornow = 1;
            m = sp->number_of_unknowns - 1;
            sp->depfixtype = 1;
        }
/*
 *      no need for z derivatives when depth is fixed
 */
        if (fixdepthfornow)
            iszderiv = 0;
/*
 *      set delta, esaz and seaz for each phase
 */
        calc_delaz(sp, p, 0);
/*
 *      Reidentify phases iff depth crosses Moho or Conrad
 */
        if ((sp->depth > moho && prev_depth <= moho) ||
            (sp->depth < moho && prev_depth >= moho) ||
            (sp->depth > conrad && prev_depth <= conrad) ||
            (sp->depth < conrad && prev_depth >= conrad)) {
            if (verbose) {
                fprintf(logfp, "    depth: %.2f prev_depth: %.2f; ",
                        sp->depth, prev_depth);
                fprintf(logfp, "reidentifying phases\n");
            }
            ispchange = reidentify_pha(sp, rdindx, p, ec, tt_tables, topo);
            mark_duplicates(sp, p, ec, tt_tables, topo);
        }
/*
 *      get residuals w.r.t. current solution
 */
        if (getresids(sp, rdindx, p, ec, tt_tables, topo, iszderiv, &dpok,
                      &ndef, &ischanged, iter, ispchange, nd, &nunp, phundef,
                      dcov, w))
            break;
        if (ndef <= sp->number_of_unknowns) {
            loss = 100. * ((double)(nd - ndef) / (double)nd);
            if (verbose) {
                fprintf(logfp, "Insufficient number (%d) of phases left; ",
                        ndef);
                fprintf(logfp, "%.0f%% phase loss!\n", loss);
            }
            errorcode = 7;
            break;
        }
        for (j = 2; j > 0; j--) nds[j] = nds[j-1];
        nds[0] = ndef;
/*
 *      initial memory allocations
 */
        if (iter == 0) {
            nd = ndef;
            prank = ndef;
            g = alloc_matrix(nd, 4);
            v = alloc_matrix(4, 4);
            d = (double *)calloc(nd, sizeof(double));
            if ((sv = (double *)calloc(4, sizeof(double))) == NULL) {
                fprintf(logfp, "locate_event: cannot allocate memory!\n");
                fprintf(errfp, "locate_event: cannot allocate memory!\n");
                errorcode = 1;
                break;
            }
/*
 *          account for correlated error structure
 */
            if (do_correlated_errors) {
/*
 *              construct data covariance matrix
 */
                if ((dcov = data_covariance_matrix(nsta, sp->numphas, nd, p,
                                 stalist, distmatrix, variogramp)) == NULL)
                    break;
/*
 *              projection matrix
 */
                if ((w = alloc_matrix(nd, nd)) == NULL) break;
                if (projection_matrix(sp->numphas, p, nd, 95., dcov, w,
                                      &prank, nunp, phundef, 1))
                    break;
            }
        }
/*
 *      rest of the iterations:
 *          check if set of time-defining phases or their names were changed
 */
        else if (ispchange) {
/*
 *          change in defining phase names:
 *              reallocate memory for G and d
 */
            if (nd != ndef) {
                free_matrix(g);
                Free(d);
                isconv = 0;
                nd = ndef;
                g = alloc_matrix(nd, 4);
                if ((d = (double *)calloc(nd, sizeof(double))) == NULL)
                    break;
            }
            if (do_correlated_errors) {
/*
 *              recalculate the data covariance and projection matrices
 */
                if (verbose) {
                    fprintf(logfp, "    Changes in defining phasenames, ");
                    fprintf(logfp, "recalculating projection matrix\n");
                }
                free_matrix(dcov);
                free_matrix(w);
                if ((dcov = data_covariance_matrix(nsta, sp->numphas, nd, p,
                                 stalist, distmatrix, variogramp)) == NULL)
                    break;
/*
 *              projection matrix
 */
                if ((w = alloc_matrix(nd, nd)) == NULL) break;
                if (projection_matrix(sp->numphas, p, nd, 95., dcov, w,
                                      &prank, nunp, phundef, ispchange))
                    break;
            }
            else
                prank = ndef;
        }
        else if (ischanged) {
/*
 *          change in number of defining phases:
 *              recalculate the projection matrix for phases that were changed
 */
            isconv = 0;
            nd = ndef;
            if (do_correlated_errors) {
                if (verbose) {
                    fprintf(logfp, "    Changes in defining phasenames, ");
                    fprintf(logfp, "recalculating projection matrix\n");
                }
                if (projection_matrix(sp->numphas, p, nd, 95., dcov, w,
                                      &prank, nunp, phundef, ispchange))
                    break;
            }
            else
                prank = ndef;
        }
        if (prank <= sp->number_of_unknowns) {
            fprintf(logfp, "Insufficient number of independent phases (%d)!\n",
                    prank);
            errorcode = 6;
            break;
        }
/*
 *      build G matrix and d vector
 */
        urms = build_gd(nd, sp, p, fixdepthfornow, g, d);
        if (do_correlated_errors) {
/*
 *          project Gm = d into eigensystem
 */
            if (project_gd(nd, m, g, d, w, &dnorm, &wrms))
                break;
        }
        else {
/*
 *          independent observations: weight Gm = d by measurement errors
 */
            weight_gd(nd, m, sp->numphas, p, g, d, &dnorm, &wrms);
        }
/*
 *      finish if convergent or divergent solution
 *          for the last iteration we only need urms and wrms
 */
        if (isconv || isdiv)
            break;
/*
 *      transpose(G) * d matrix norm
 */
        gtdnorm = 0.;
        for (i = 0; i < nd; i++) {
            gtd = 0.0;
            for (j = 0; j < m; j++)
                gtd += g[i][j] * d[i];
            gtdnorm += gtd * gtd;
        }
/*
 *      SVD of G (G is overwritten by U matrix!)
 */
        if (svd_decompose(nd, m, g, sv, v))
            break;
        for (j = 0; j < m; j++) svundamped[j] = sv[j];
/*
 *      condition number, G matrix norm, rank and convergence test value
 */
        svth = svd_threshold(nd, m, sv);
        nr = svd_rank(nd, m, sv, svth);
        gnorm = svd_norm(m, sv, svth, &cond);
        if (nr < m) {
            fprintf(logfp, "Singular G matrix (%d < %d)!\n", nr, m);
            fprintf(errfp, "Singular G matrix (%d < %d)!\n", nr, m);
            errorcode = 9;
            break;
        }
        if (cond > 30000.) {
            fprintf(logfp, "Abnormally ill-conditioned problem (cond=%.0f)!\n",
                    cond);
            fprintf(errfp, "Abnormally ill-conditioned problem (cond=%.0f)!\n",
                    cond);
            errorcode = 10;
            break;
        }
        cnvgtst = convtestval(gtdnorm, gnorm, dnorm);
/*
 *      If damping is enabled, apply damping if condition number is large.
 *      Apply of 1% largest singular value for moderately ill-conditioned,
 *               5% for more severely ill-conditioned and
 *              10% for highly ill-conditioned problems.
 */
        if (allow_damping && cond > 30.) {
            damp = 0.01;
            if (cond > 300.)  damp = 0.05;
            if (cond > 3000.) damp = 0.1;
            for (j = 1; j < nr; j++)
                sv[j] += sv[0] * damp;
            if (verbose) {
                fprintf(logfp, "    Large condition number (%.3f): ", cond);
                fprintf(logfp, "%.0f%% damping is applied.\n", 100. * damp);
            }
        }
/*
 *      solve Gm = d
 */
        if (svd_solve(nd, m, g, sv, v, d, sol, svth))
            break;
/*
 *      model norm
 */
        for (mnorm = 0., j = 0; j < m; j++)
            mnorm += sol[j] * sol[j];
        mnorm = Sqrt(mnorm);
/*
 *      scale down hypocenter perturbations if they are very large
 */
        dmax = 1000.;
        if (mnorm > dmax) {
            scale = dmax / mnorm;
            for (j = 0; j < m; j++)
                sol[j] *= scale;
            mnorm = dmax;
            if (verbose) {
                fprintf(logfp, "    Large perturbation: ");
                fprintf(logfp, "%.g scaling is applied.\n", scale);
            }
        }
/*
 *      convergence test
 */
        for (j = 2; j > 0; j--) {
            modelnorm[j] = modelnorm[j-1];
            convgtest[j] = convgtest[j-1];
        }
        modelnorm[0] = mnorm;
        convgtest[0] = cnvgtst;
        if (iter > min_iter - 1)
            isconv = convergence_test(iter, m, nds, sol, oldsol, wrms,
                            modelnorm, convgtest, &oldcvgtst, &step, &isdiv);
/*
 *      update hypocentre coordinates
 */
        prev_depth = sp->depth;
        if (verbose) {
            fprintf(logfp, "    iteration = %d: ", iter);
            if     (isconv) fprintf(logfp, "    converged!\n");
            else if (isdiv) fprintf(logfp, "    diverged!\n");
            else            fprintf(logfp, "\n");
            fprintf(logfp, "    ||Gt*d|| = %.5f ||G|| = %.5f ", gtdnorm, gnorm);
            fprintf(logfp, "||d|| = %.5f ||m|| = %.5f\n", dnorm, mnorm);
            fprintf(logfp, "    convgtst = %.5f condition number = %.3f\n",
                    cnvgtst, cond);
            fprintf(logfp, "    eigenvalues: ");
            for (i = 0; i < m; i++) fprintf(logfp, "%g ", sv[i]);
            fprintf(logfp, "\n    unweighted RMS residual = %8.4f\n", urms);
            fprintf(logfp, "      weighted RMS residual = %8.4f\n", wrms);
            fprintf(logfp, "    ndef = %d rank = %d m = %d ischanged = %d\n",
                            nd, prank, m, ischanged);
        }
        i = 0;
        if (verbose) fprintf(logfp, "    ");
        if (!sp->timfix) {
            if (verbose) fprintf(logfp, "dOT = %g ", sol[i]);
            torg += sol[i++];
            sp->time = torg + toffset;
        }
        if (!sp->epifix) {
            azim = RAD_TO_DEG * atan2(sol[i], sol[i+1]);
            delta = Sqrt(sol[i] * sol[i] + sol[i+1] * sol[i+1]);
            delta = RAD_TO_DEG * (delta / (EARTH_RADIUS - sp->depth));
            deltaloc(sp->lat, sp->lon, delta, azim, &sp->lat, &sp->lon);
            if (verbose)
                fprintf(logfp, "dx = %g dy = %g ", sol[i], sol[i+1]);
            i += 2;
        }
        if (!fixdepthfornow) {
            if (verbose) fprintf(logfp, "dz = %g ", -sol[i]);
            sp->depth -= sol[i];
        }
        if (verbose) {
            fprintf(logfp, "\n");
            print_sol(sp, 0);
        }
        if (verbose > 1)
            print_defining_pha(sp->numphas, p);
        if (verbose)
            fprintf(logfp, "iteration = %d (%.4f) done\n", iter, secs(&t0));
    }
/*
 *
 *  end of iteration loop
 *
 */
    if (verbose)
        fprintf(logfp, "    end of iteration loop (%.4f)\n", secs(&t0));
/*
 *  Sort phase structures so that they ordered by delta, prista, rdid, time
 */
    sort_phaserec_db(sp->numphas, p);
    readings(sp->numphas, sp->nreading, p, rdindx);
    if (verbose > 3)
        print_pha(sp->numphas, p);
/*
 *
 *  max number of iterations reached
 *
 */
    if (iter >= max_iter) {
        fprintf(logfp, "    maximum number of iterations is reached!\n");
        errorcode = 8;
        isdiv = 1;
    }
/*
 *
 *  convergent solution
 *      calculate model covariance matrix
 *
 */
    else if (isconv) {
        fprintf(logfp, "    convergent solution after %d iterations\n", iter);
        iserr = 0;
        sp->urms = urms;
        sp->wrms = wrms;
        sp->prank = prank;
        sp->ndef = nd;
        if (!sp->depfixtype && sp->depth < DEPSILON) {
            fixdepthfornow = 1;
            sp->depfixtype = 1;
            sp->depth = 0.;
            m--;
        }
        if (!sp->depfixtype && sp->depth > max_depth_km - DEPSILON) {
            fixdepthfornow = 1;
            sp->depfixtype = 1;
            sp->depth = max_depth_km;
            m--;
        }
        sp->number_of_unknowns = m;
        sp->depfix = fixdepthfornow;
        svd_model_covariance_matrix(m, svth, svundamped, v, mcov);
        if (!sp->timfix) {
            sp->covar[0][0] = mcov[0][0];                /* stt */
            if (!sp->epifix) {
                sp->covar[0][1] = mcov[0][1];            /* stx */
                sp->covar[0][2] = mcov[0][2];            /* sty */
                sp->covar[1][0] = mcov[1][0];            /* sxt */
                sp->covar[1][1] = mcov[1][1];            /* sxx */
                sp->covar[1][2] = mcov[1][2];            /* sxy */
                sp->covar[2][0] = mcov[2][0];            /* syt */
                sp->covar[2][1] = mcov[2][1];            /* syx */
                sp->covar[2][2] = mcov[2][2];            /* syy */
                if (!fixdepthfornow) {
                    sp->covar[0][3] = mcov[0][3];        /* stz */
                    sp->covar[1][3] = mcov[1][3];        /* sxz */
                    sp->covar[2][3] = mcov[2][3];        /* syz */
                    sp->covar[3][0] = mcov[3][0];        /* szt */
                    sp->covar[3][1] = mcov[3][1];        /* szx */
                    sp->covar[3][2] = mcov[3][2];        /* szy */
                    sp->covar[3][3] = mcov[3][3];        /* szz */
                }
            }
            else {
                if (!fixdepthfornow) {
                    sp->covar[0][3] = mcov[0][1];        /* stz */
                    sp->covar[3][0] = mcov[1][0];        /* szt */
                    sp->covar[3][3] = mcov[1][1];        /* szz */
                }
            }
        }
        else {
            if (!sp->epifix) {
                sp->covar[1][1] = mcov[0][0];            /* sxx */
                sp->covar[1][2] = mcov[0][1];            /* sxy */
                sp->covar[2][1] = mcov[1][0];            /* syx */
                sp->covar[2][2] = mcov[1][1];            /* syy */
                if (!fixdepthfornow) {
                    sp->covar[1][3] = mcov[0][2];        /* sxz */
                    sp->covar[2][3] = mcov[1][2];        /* syz */
                    sp->covar[3][1] = mcov[2][0];        /* szx */
                    sp->covar[3][2] = mcov[2][1];        /* szy */
                    sp->covar[3][3] = mcov[2][2];        /* szz */
                }
            }
            else {
                if (!fixdepthfornow)
                    sp->covar[3][3] = mcov[0][0];        /* szz */
            }
        }
/*
 *      location uncertainties
 */
        calc_error(sp, p);
    }
/*
 *
 *  divergent solution
 *
 */
    else if (isdiv) {
        fprintf(logfp, "    divergent solution\n");
        errorcode = 4;
    }
/*
 *  abnormal exit
 */
    else {
        fprintf(logfp, "    abnormal exit from iteration loop!\n");
        fprintf(errfp, "    abnormal exit from iteration loop!\n");
        isdiv = 1;
    }
    sp->converged = isconv;
    sp->diverging = isdiv;
/*
 *  free memory allocated to various arrays
 */
    if (do_correlated_errors) {
        free_matrix(w);
        free_matrix(dcov);
    }
    Free(sv);
    Free(d);
    free_matrix(v);
    free_matrix(g);
    return iserr;
}

/*
 *  Title:
 *     getndef
 *  Synopsis:
 *     Finds number of defining phases and the earliest arrival time.
 *  Input Arguments:
 *     numphas   - number of associated phases
 *     p[]       - array of phase structures
 *     nsta      - number of distinct stations
 *     stalist[] - array of starec structures
 *  Output Arguments:
 *     toffset   - earliest arrival time
 *  Return:
 *     nd - number of defining phases at valid stations or -1 on error
 *  Called by:
 *     locate_event
 *  Calls:
 *     get_sta_index
 */
static int getndef(int numphas, PHAREC p[], int nsta, STAREC stalist[],
                   double *toffset)
{
    int i, nd = 0, sind = -1;
    double toff = 1e+32;
    for (i = 0; i < numphas; i++) {
        if (!p[i].timedef) continue;
        if ((sind = get_sta_index(nsta, stalist, p[i].prista)) < 0) {
            fprintf(logfp, "getndef: %-6s invalid station!\n", p[i].prista);
            fprintf(errfp, "getndef: %-6s invalid station!\n", p[i].prista);
            errorcode = 11;
            return -1;
        }
        p[i].prevtimedef = p[i].timedef;
        strcpy(p[i].prevphase, p[i].phase);
        if (p[i].time < toff)
            toff = p[i].time;
        nd++;
    }
    *toffset = toff;
    return nd;
}

/*
 *  Title:
 *     getphases
 *  Synopsis:
 *     populates PHASELIST for defining phases
 *     PHASELIST contains:
 *          phase name
 *          number of observations for this phase
 *          permutation vector that renders the data covariance matrix
 *              block-diagonal (phase by phase)
 *     returns number of distinct defining phases
 *  Input Arguments:
 *     numphas - number of associated phases
 *     p[]     - array of phase structures
 *  Output Arguments:
 *     plist - PHASELIST structure
 *  Return:
 *     nphases or 0 on error
 *  Called by:
 *     projection_matrix
 */
int getphases(int numphas, PHAREC p[], PHASELIST plist[])
{
    int i, j, k, m, nphases = 0, isfound = 0;
    for (j = 0; j < MAXTTPHA; j++) {
        strcpy(plist[j].phase, "");
        plist[j].n = 0;
        plist[j].ind = (int *)NULL;
    }
/*
 *  get number of defining observations for each phase
 */
    for (i = 0; i < numphas; i++) {
        if (!p[i].timedef) continue;
/*
 *      find phase in plist
 */
        isfound = 0;
        for (j = 0; j < nphases; j++) {
            if (streq(p[i].phase, plist[j].phase)) {
                plist[j].n++;
                isfound = 1;
                break;
            }
        }
/*
 *      new phase; add it to plist
 */
        if (!isfound) {
            strcpy(plist[j].phase, p[i].phase);
            plist[j].n = 1;
            nphases++;
        }
    }
/*
 *  allocate memory and build permutation vectors
 */
    for (j = 0; j < nphases; j++) {
        if ((plist[j].ind = (int *)calloc(plist[j].n, sizeof(int))) == NULL) {
            freephaselist(nphases, plist);
            fprintf(logfp, "getphases: cannot allocate memory\n");
            fprintf(errfp, "getphases: cannot allocate memory\n");
            errorcode = 1;
            return 0;
        }
        m = -1;
        for (k = 0, i = 0; i < numphas; i++) {
            if (p[i].timedef) m++;
            else continue;
            if (streq(p[i].phase, plist[j].phase))
                plist[j].ind[k++] = m;
        }
    }
    return nphases;
}

/*
 *  Title:
 *     freephaselist
 *  Synopsis:
 *     frees memory allocated to PHASELIST structure
 *  Input Arguments:
 *     nphases number of distinct defining phases
 *     plist - PHASELIST structure
 *  Called by:
 *     projection_matrix
 */
void freephaselist(int nphases, PHASELIST plist[])
{
    int j;
    for (j = 0; j < nphases; j++) {
        strcpy(plist[j].phase, "");
        plist[j].n = 0;
        Free(plist[j].ind);
    }
}

/*
 *  Title:
 *     getresids
 *  Synopsis:
 *     Sets residuals for defining phases.
 *     Flags first arriving P and remove orphan depth phases
 *     Makes a phase non-defining if its residual is larger than
 *         sigmathres times the prior measurement error and
 *         deletes corresponding row and column in the data covariance and
 *         projection matrices.
 *  Input Arguments:
 *     sp        - pointer to current solution
 *     p[]       - array of phase structures
 *     ec        - pointer to ellipticity correction coefficient structure
 *     tt_tables - pointer to travel-time tables
 *     topo      - ETOPO bathymetry/elevation matrix
 *     iszderiv  - calculate dtdh [0/1]?
 *     iter      - iteration number
 *     ispchange - change in phase names?
 *     prevndef  - number of defining phases from previous iteration
 *     dcov      - data covariance matrix from previous iteration
 *     w         - projection matrix from previous iteration
 *  Output Arguments:
 *     has_depdpres - do we have depth-phase depth resolution?
 *     ndef      - number of defining phases
 *     ischanged - change in the set of defining phases? [0/1]
 *     nunp      - number of distinct phases made non-defining
 *     phundef   - list of distinct phases made non-defining
 *     dcov      - data covariance matrix
 *     w         - projection matrix
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     locate_event
 *  Calls:
 *     depth_phase_check, calc_resid
 */
static int getresids(SOLREC *sp, READING *rdindx, PHAREC p[], EC_COEF *ec,
                     TT_TABLE *tt_tables, short int **topo, int iszderiv,
                     int *has_depdpres, int *ndef, int *ischanged, int iter,
                     int ispchange, int prevndef, int *nunp, char **phundef,
                     double **dcov, double **w)
{
    int i, j, k = 0, m = 0, kp = 0, nd = 0, nund = 0, isdiff = 0, isfound = 0;
    extern double sigmathres;                           /* from config file */
    double thres = 0.;
/*
 *  flag first arriving P and remove orphan depth phases
 */
    k = depth_phase_check(sp, rdindx, p, 0);
    *has_depdpres = k;
/*
 *  set ttime, residual, dtdh, and dtdd for defining phases
 */
    if (calc_resid(sp, p, "use", ec, tt_tables, topo, iszderiv))
        return 1;
/*
 *  see if set of time defining phases has changed
 */
    for (i = 0; i < sp->numphas; i++) {
        if (p[i].timedef) {
            thres = sigmathres * p[i].measerr;
/*
 *          make phase non-defining if its residual is larger than
 *          sigmathres times the prior measurement error
 */
            if (fabs(p[i].resid) > thres)
                p[i].timedef = 0;
            else nd++;
        }
        if (p[i].timedef != p[i].prevtimedef) {
            isdiff = 1;
            nund++;
            if (verbose > 2)
                fprintf(logfp, "        %-6s %-8s %10.3f made non-defining\n",
                                p[i].sta, p[i].phase, p[i].resid);
/*
 *          delete corresponding row and column in dcov and w
 */
            if (iter && !ispchange && do_correlated_errors) {
                isfound = 0;
                for (j = 0; j < kp; j++)
                    if (streq(phundef[j], p[i].phase)) isfound = 1;
                if (!isfound) strcpy(phundef[kp++], p[i].phase);
/*
 *             index of phase in dcov and w
 */
                k = p[i].covindex;
                for (j = k; j < prevndef - 1; j++) {
                    for (m = 0; m < prevndef; m++) {
                        dcov[j][m] = dcov[j+1][m];
                        w[j][m] = w[j+1][m];
                    }
                }
                for (j = k; j < prevndef - 1; j++) {
                    for (m = 0; m < prevndef; m++) {
                        dcov[m][j] = dcov[m][j+1];
                        w[m][j] = w[m][j+1];
                    }
                }
            }
        }
        p[i].prevtimedef = p[i].timedef;
    }
    if (verbose)
        fprintf(logfp, "    getresids: %d phases made non-defining\n", nund);
    *ndef = nd;
    *nunp = kp;
    *ischanged = isdiff;
    return 0;
}

/*
 *  Title:
 *     build_gd
 *  Synopsis:
 *     Builds G matrix and d vector for equation Gm = d.
 *     G is the matrix of partial derivates of travel-times,
 *     d is the vector of residuals.
 *  Input arguments
 *     ndef - number of defining phases
 *     sp   - pointer to current solution
 *     p[]  - array of phase structures
 *     fixdepthfornow - fix depth for this iteration?
 *  Output arguments
 *     g    - G matrix G(N x 4)
 *     d    - residual vector d(N)
 *  Return:
 *     urms - unweighted rms residual
 *  Called by:
 *     locate_event
 */
static double build_gd(int ndef, SOLREC *sp, PHAREC p[], int fixdepthfornow,
                       double **g, double *d)
{
    int i, j, k, im = 0;
    double urms = 0., depthcorr = 0., esaz = 0.;
	depthcorr = DEG_TO_RAD * (EARTH_RADIUS - sp->depth);
/*
 *  build G matrix and d vector
 */
    if (verbose > 2)
        fprintf(logfp, "        G matrix and d vector:\n");
    for (k = 0, i = 0; i < sp->numphas; i++) {
        if (!p[i].timedef) continue;
        for (j = 0; j < 4; j++) g[k][j] = 0.;
/*
 *      G matrix of partial derivates of travel-times
 */
        im = 0;
        if (!sp->timfix)       /* Time */
            g[k][im++] = 1.;
        if (!sp->epifix) {     /* E, N */
            esaz = DEG_TO_RAD * p[i].esaz;
            g[k][im++] = -(p[i].dtdd / depthcorr) * sin(esaz);
            g[k][im++] = -(p[i].dtdd / depthcorr) * cos(esaz);
        }
        if (!fixdepthfornow)   /* Up */
            g[k][im++] = -p[i].dtdh;
/*
 *      d vector and unweighted rms residual
 */
        d[k] = p[i].resid;
        urms += d[k] * d[k];
        if (verbose > 2) {
            fprintf(logfp, "          %6d %-6s %-9s",
                    k, p[i].prista, p[i].phase);
            for (j = 0; j < im; j++) fprintf(logfp, "%12.6f ", g[k][j]);
            fprintf(logfp, " , %12.6f\n", d[k]);
        }
        k++;
    }
    urms = Sqrt(urms / (double)ndef);
    return urms;
}

/*
 *  Title:
 *     project_gd
 *  Synopsis:
 *     Projects G matrix and d vector into eigensystem.
 *  Input arguments:
 *     ndef  - number of defining phases
 *     m     - number of model parameters
 *     g     - G matrix G(N x M)
 *     d     - residual vector d(N)
 *     w     - W projection matrix W(N x N)
 *  Output arguments:
 *     g     - projected G matrix G(N x M)
 *     d     - projected residual vector d(N)
 *     dnorm - data norm (sum of squares of weighted residuals)
 *     wrms  - weighted rms residual
 *  Returns:
 *     0/1 on success/error
 *  Called by:
 *     locate_event
 *  Calls:
 *     WxG
 */
static int project_gd(int ndef, int m, double **g, double *d, double **w,
                      double *dnorm, double *wrms)
{
    int i, j, k;
    double *temp = (double *)NULL;
    double wssq = 0.;
/*
 *  allocate memory for temporary storage
 */
    if ((temp = (double *)calloc(ndef, sizeof(double))) == NULL) {
        fprintf(logfp, "project_gd: cannot allocate memory\n");
        fprintf(errfp, "project_gd: cannot allocate memory\n");
        errorcode = 1;
        return 1;
    }
/*
 *  WG(NxM) = W(NxN) * G(NxM)
 */
#ifdef WITH_GCD
/*
 *  use GCD (Mac OS) to parallelize the matrix multiplication
 *  each model dimension is processed concurrently
 */
    if (ndef > 100) {
        dispatch_apply(m, dispatch_get_global_queue(0, 0), ^(size_t j){
            WxG((int)j, ndef, w, g);
        });
    }
    else {
        for (j = 0; j < m; j++) {
            WxG(j, ndef, w, g);
        }
    }
#else
/*
 *  single core
 */
    for (j = 0; j < m; j++) {
        WxG(j, ndef, w, g);
    }
#endif
    if (errorcode)
        return 1;
/*
 *  Wd(N) = W(NxN) * d(N) and sum of squares of weighted residuals
 */
    for (i = 0; i < ndef; i++) {
        temp[i] = 0.;
        for (k = 0; k < ndef; k++)
            temp[i] += w[i][k] * d[k];
        if (fabs(temp[i]) < ZERO_TOL) temp[i] = 0.;
    }
    wssq = 0.;
    for (i = 0; i < ndef; i++) {
        d[i] = temp[i];
        wssq += d[i] * d[i];
    }
    Free(temp);
    *dnorm = wssq;
    *wrms = Sqrt(wssq / (double)ndef);
    if (verbose > 2) {
        fprintf(logfp, "        WG(%d x %d) matrix and Wd vector:\n", ndef, m);
        for (i = 0; i < ndef; i++) {
            fprintf(logfp, "          %6d ", i);
            for (j = 0; j < m; j++) fprintf(logfp, "%12.6f ", g[i][j]);
            fprintf(logfp, " , %12.6f\n", d[i]);
        }
    }
    return 0;
}

/*
 *  Title:
 *     WxG
 *  Synopsis:
 *     W * G matrix multiplication
 *  Input arguments:
 *     j     - model dimension index
 *     ndef  - number of defining phases
 *     w     - W projection matrix W(N x N)
 *     g     - G matrix G(N x M)
 *  Output arguments:
 *     g     - projected G matrix G(N x M)
 *  Returns:
 *     0/1 on success/error
 *  Called by:
 *     project_gd
 */
static int WxG(int j, int ndef, double **w, double **g)
{
    int i, k;
    double *temp = (double *)NULL;
    if ((temp = (double *)calloc(ndef, sizeof(double))) == NULL) {
        fprintf(logfp, "WxG: cannot allocate memory\n");
        fprintf(errfp, "WxG: cannot allocate memory\n");
        errorcode = 1;
        return 1;
    }
    for (k = 0; k < ndef; k++)
        temp[k] = g[k][j];
    for (i = 0; i < ndef; i++) {
        g[i][j] = 0.;
        for (k = 0; k < ndef; k++)
            g[i][j] += w[i][k] * temp[k];
        if (fabs(g[i][j]) < ZERO_TOL) g[i][j] = 0.;
    }
    Free(temp);
    return 0;
}


/*
 *  Title:
 *     weight_gd
 *  Synopsis:
 *     Independence assumption: weight Gm = d by measurement errors.
 *  Input arguments
 *     ndef    - number of defining phases
 *     m       - number of model parameters
 *     numphas - number of associated phases
 *     p[]     - array of phase structures
 *     g       - G matrix G(N x M)
 *     d       - residual vector d(N)
 *  Output arguments
 *     g     - weighted G matrix G(N x M)
 *     d     - weighted residual vector d(N)
 *     dnorm - data norm (sum of squares of weighted residuals)
 *     wrms  - weighted rms residual
 *  Called by:
 *     locate_event
 */
static void weight_gd(int ndef, int m, int numphas, PHAREC p[],
                      double **g, double *d, double *dnorm, double *wrms)
{
    int i, j, k;
    double wssq = 0., weight = 0.;
    for (k = 0, i = 0; i < numphas; i++) {
        if (!p[i].timedef) continue;
        if (p[i].measerr < DEPSILON)
            weight = 1.;
        else
            weight = 1. / p[i].measerr;
        for (j = 0; j < m; j++)
            g[k][j] *= weight;
        d[k] *= weight;
        wssq += d[k] * d[k];
        k++;
    }
    *dnorm = wssq;
    *wrms = Sqrt(wssq / (double)ndef);
    if (verbose > 2) {
        fprintf(logfp, "        WG(%d x %d) matrix and Wd vector:\n", ndef, m);
        for (i = 0; i < ndef; i++) {
            fprintf(logfp, "          %6d ", i);
            for (j = 0; j < m; j++) fprintf(logfp, "%12.6f ", g[i][j]);
            fprintf(logfp, " , %12.6f\n", d[i]);
        }
    }
}

/*
 *  Title:
 *     convtestval
 *  Synopsis:
 *     Convergence test value of Paige and Saunders (1982)
 *
 *     Paige, C. and Saunders, M., 1982,
 *         LSQR: An Algorithm for Sparse Linear Equations and
 *         Sparse Least Squares,
 *         ACM Trans. Math. Soft. 8, 43-71.
 *
 *               ||transpose(G) * d||
 *     cvgtst = ----------------------
 *                  ||G|| * ||d||
 *
 *  Input arguments:
 *     gtdnorm  - ||transpose(G) * d||
 *     gnorm    - G matrix norm (Sum(sv^2))
 *     dnorm    - d vector norm (Sum(d^2))
 *  Return:
 *     cnvgtst  - Paige-Saunders convergence test number
 *  Called by:
 *     locate_event
 */
static double convtestval(double gtdnorm, double gnorm, double dnorm)
{
    double cnvgtst = 0., gd = 0.;
    gd = gnorm * dnorm;
    if (gtdnorm > DEPSILON && gd < DEPSILON)
        cnvgtst = 999.;
    else
        cnvgtst = gtdnorm / gd;
    return cnvgtst;
}

/*
 *  Title:
 *     convergence_test
 *  Synopsis:
 *     Convergence/divergence is decided based on
 *         the Paige-Saunder convergence test value and
 *         the history of model and data norms.
 *  Input Arguments:
 *     iter        - iteration number
 *     m           - number of model parameters
 *     nds[]       - current and past number of defining phases
 *     sol[]       - current solution vector
 *     oldsol[]    - previous solution vector
 *     wrms        - weighted RMS residual
 *     modelnorm[] - current and past model norms
 *     convgtest[] - current and past convergence test values
 *     oldcvgtst   - previous convergence test value
 *     step        - step length
 *  Output Arguments:
 *     oldsol[]    - previous solution vector
 *     oldcvgtst   - previous convergence test value
 *     step        - step length
 *     isdiv       - divergent solution [0/1]
 *  Returns:
 *     isconv - convergent solution [0/1]
 *  Called by:
 *     locate_event
 */
static int convergence_test(int iter, int m, int *nds, double *sol,
                            double *oldsol, double wrms, double *modelnorm,
                            double *convgtest, double *oldcvgtst,
                            double *step, int *isdiv)
{
    double dm01 = 0., dm12 = 0., dc01 = 0., dc12 = 0.;
    double sc = *step, oldcvg = 0.;
    int i, convergent = 0, divergent = 0;
    oldcvg = *oldcvgtst;
    if (modelnorm[0] > 0. && convgtest[0] > 0.) {
/*
 *      indicators of increasing/decreasing model norms and convergence tests
 */
        if (modelnorm[1] <= 0. || modelnorm[2] <= 0.)
            dm01 = dm12 = 1.05;
        else {
            dm01 = modelnorm[0] / modelnorm[1];
            dm12 = modelnorm[1] / modelnorm[2];
        }
        if (convgtest[1] <= 0. || convgtest[2] <= 0.)
            dc01 = convgtest[0];
        else {
            dc01 = convgtest[0] / convgtest[1];
            dc12 = convgtest[1] / convgtest[2];
            dc01 = fabs(dc12 - dc01);
        }
        dc12 = fabs(convgtest[0] - convgtest[2]);
/*
 *      divergent solution if increasing model norm
 */
        if (dm12 > 1.1 && dm01 > dm12 &&
            iter > min_iter + 2 && modelnorm[0] > 500)
            divergent = 1;
/*
 *      convergent solution if vanishing
 *      convergence test value or model norm or weighted RMS residual
 */
        else if (nds[0] == nds[1] &&
                (convgtest[0] < CONV_TOL || modelnorm[0] < 0.1 || wrms < 0.01))
            convergent = 1;
/*
 *      convergent solution if vanishing convergence test value
 */
        else if ((convgtest[0] < 1.01 * oldcvg && convgtest[0] < CONV_TOL)
                 || (iter > 3 * max_iter / 4
                     && (convgtest[0] < sqrt(CONV_TOL) ||
                         dc01 < CONV_TOL || dc12 < sqrt(CONV_TOL))
                    )
                )
            convergent = 1;
    }
    else
        convergent = 1;
    if (iter == max_iter - 1)
        convergent = 0;
/*
 *  Apply step-length weighting if convergence test value is increasing.
 *  Steps are applied in half-lengths of previous solution vector
 */
    if (iter > min_iter + 2 &&
        (convgtest[0] > *oldcvgtst || convgtest[0] - convgtest[2] == 0.) &&
        sc > 0.05) {
        sc *= 0.5;
        if (sc != 0.5) {
            for (i = 0; i < m; i++) {
                if (fabs(oldsol[i]) < ZERO_TOL) oldsol[i] = sol[i];
                sol[i] = sc * oldsol[i];
            }
        }
        else {
            for (i = 0; i < m; i++) {
                sol[i] = sc * sol[i];
                oldsol[i] = sol[i];
            }
        }
    }
    else {
        sc = 1;
        *oldcvgtst = convgtest[0];
    }
    *step = sc;
    *isdiv = divergent;
    return convergent;
}

/*
 *  Title:
 *     readings
 *  Synopsis:
 *     records starting index and number of phases in a reading
 *  Input Arguments:
 *     numphas  - number of associated phases
 *     nreading - number of readings
 *     p        - array of phase structures
 *  Output Arguments:
 *     rdindx   - array of reading structures
 *  Called by:
 *     eventloc, locate_event, na_search
 */
void readings(int numphas, int nreading, PHAREC p[], READING *rdindx)
{
    int i, j = 0, rdid;
    for (i = 0; i < nreading; i++) {
        rdindx[i].start = j;
        rdindx[i].npha = 0;
        rdid = p[j].rdid;
        for (; j < numphas; j++) {
            if (p[j].rdid != rdid)
                break;
            rdindx[i].npha++;
        }
    }
}
