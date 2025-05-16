#include "iscloc.h"
extern int verbose;
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;

/*
 * Functions:
 *    print_sol
 *    print_hyp
 *    print_pha
 *    print_defining_pha
 */

/*
 *  Title:
 *      print_sol
 *  Synopsis:
 *      Prints current solution to the logfile
 *  Input Arguments:
 *      sp  - pointer to current solution.
 *      grn - geographic region number. gregname is printed if grn > 0.
 *  Calls:
 *      gregion, human_time
 *  Called by:
 *      eventloc, locate_event, na_search
 */
void print_sol(SOLREC *sp, int grn)
{
    char timestr[25], gregname[255];
    if (grn) {
        gregion(grn, gregname);
        fprintf(logfp, "%s\n", gregname);
    }
    human_time(timestr, sp->time);
    fprintf(logfp, "    OT = %s ", timestr);
    if (sp->error[0] != NULLVAL)
        fprintf(logfp, "+/- %.3f[s] ", sp->error[0]);
    if (sp->error[1] != NULLVAL)
        fprintf(logfp, "Lat = %.3f +/- %.3f[deg] ", sp->lat, sp->error[1]);
    else
        fprintf(logfp, "Lat = %.3f ", sp->lat);
    if (sp->error[2] != NULLVAL)
        fprintf(logfp, "Lon = %.3f +/- %.3f[deg] ", sp->lon, sp->error[2]);
    else
        fprintf(logfp, "Lon = %.3f ", sp->lon);
    if (sp->error[3] != NULLVAL)
        fprintf(logfp, "Depth = %.1f +/- %.1f[km]\n", sp->depth, sp->error[3]);
    else
        fprintf(logfp, "Depth = %.1f\n", sp->depth);
}

/*
 *  Title:
 *      print_hyp
 *  Synopsis:
 *      Prints hypocentres for an event to the logfile
 *  Input Arguments:
 *      ep  - pointer to event record
 *      h[] - array of hypocentre records.
 *  Calls:
 *      human_time
 *  Called by:
 *      eventloc
 */
void print_hyp(EVREC *ep, HYPREC h[])
{
    char timestr[25];
    int i;
    fprintf(logfp, "numhyps=%d numphas=%d\n", ep->numhyps, ep->numphas);
    fprintf(logfp, "hypid     agency   time                     lat");
    fprintf(logfp, "     lon     depth  nsta sdef ndef nass gap  mindist ");
    fprintf(logfp, "stime sdep  smaj  score\n");
    for (i = 0; i < ep->numhyps; i++) {
        human_time(timestr, h[i].time);
        fprintf(logfp, "%9d ", h[i].hypid);
        fprintf(logfp, "%-8s %s ", h[i].agency, timestr);
        fprintf(logfp, "%7.3f %8.3f ", h[i].lat, h[i].lon);
        if (h[i].depth != NULLVAL)   fprintf(logfp, "%5.1f  ", h[i].depth);
        else                         fprintf(logfp, "%5s  ", "");
        if (h[i].nsta != NULLVAL)    fprintf(logfp, "%4d ", h[i].nsta);
        else                         fprintf(logfp, "%4s ", "");
        if (h[i].ndefsta != NULLVAL) fprintf(logfp, "%4d ", h[i].ndefsta);
        else                         fprintf(logfp, "%4s ", "");
        if (h[i].ndef != NULLVAL)    fprintf(logfp, "%4d ", h[i].ndef);
        else                         fprintf(logfp, "%4s ", "");
        if (h[i].nass != NULLVAL)    fprintf(logfp, "%4d ", h[i].nass);
        else                         fprintf(logfp, "%4s ", "");
        if (h[i].azimgap != NULLVAL) fprintf(logfp, "%5.1f ", h[i].azimgap);
        else                         fprintf(logfp, "%5s ", "");
        if (h[i].mindist != NULLVAL) fprintf(logfp, "%6.2f ", h[i].mindist);
        else                         fprintf(logfp, "%6s ", "");
        if (h[i].stime != NULLVAL)   fprintf(logfp, "%5.2f ", h[i].stime);
        else                         fprintf(logfp, "%5s ", "");
        if (h[i].sdepth != NULLVAL)  fprintf(logfp, "%5.2f ", h[i].sdepth);
        else                         fprintf(logfp, "%5s ", "");
        if (h[i].smajax != NULLVAL)  fprintf(logfp, "%5.1f ", h[i].smajax);
        else                         fprintf(logfp, "%5s ", "");
        fprintf(logfp, "%5d\n", h[i].rank);
    }
}

/*
 *  Title:
 *      print_pha
 *  Synopsis:
 *      Prints a table with all the phases for one event.
 *  Input Arguments:
 *      numphas - number of associated phases
 *      p[] - array of phase structures
 *  Calls:
 *      human_time
 *  Called by:
 *      eventloc, locate_event, synthetic, read_isf, get_pha, calc_delaz
 */
void print_pha(int numphas, PHAREC p[])
{
    int i, j;
    char timestr[25], s[255];
    fprintf(logfp, "    RDID      ARRID      STA                PCH ");
    fprintf(logfp, "REPORTED IASPEI   I TIME                     DELTA   ");
    fprintf(logfp, "ESAZ DELTIM TF RESIDUAL MTYPE        AMP  PER ACH M\n");
    for (i = 0; i < numphas; i++) {
        if (p[i].purged)       /* used by iscloc_search */
            continue;
        sprintf(s, "    %-9d", p[i].rdid);
        sprintf(s, "%s %-10s", s, p[i].arrid);
        sprintf(s, "%s %-18s", s, p[i].fdsn);
        sprintf(s, "%s %3s", s, p[i].pch);
        sprintf(s, "%s %-8s", s, p[i].rep_phase);
        sprintf(s, "%s %-8s", s, p[i].phase);
        sprintf(s, "%s %d", s, p[i].init);
        if (p[i].time != NULLVAL) {
            human_time(timestr, p[i].time);
            sprintf(s, "%s %s", s, timestr);
        }
        else sprintf(s, "%s %23s", s, "");
        sprintf(s, "%s %6.2f", s, p[i].delta);
        sprintf(s, "%s %6.2f", s, p[i].esaz);
        if (p[i].measerr != NULLVAL) sprintf(s, "%s %6.3f", s, p[i].measerr);
        else                         sprintf(s, "%s %6s", s, "");
        sprintf(s, "%s %d", s, p[i].timedef);
        sprintf(s, "%s%d", s, p[i].phase_fixed);
        if (p[i].resid != NULLVAL) sprintf(s, "%s %8.3f", s, p[i].resid);
        else                       sprintf(s, "%s %8s", s, "");
        if (p[i].numamps > 0) {
/*
 *          amplitude measurements reported with the pick
 */
            for (j = 0; j < p[i].numamps; j++) {
                if (j) fprintf(logfp, "%125s", "");
                else   fprintf(logfp, "%s ", s);
                fprintf(logfp, "%-6s ", p[i].a[j].magtype);
                if (p[i].a[j].amp != NULLVAL)
                    fprintf(logfp, "%9.1f ", p[i].a[j].amp);
                else
                    fprintf(logfp, "%9s ", "");
                if (p[i].a[j].per != NULLVAL &&
                    (tolower(p[i].a[j].magtype[1]) == 'b' ||
                     tolower(p[i].a[j].magtype[1]) == 's'))
                    fprintf(logfp, "%4.1f ", p[i].a[j].per);
                else
                    fprintf(logfp, "%4s ", "");
                fprintf(logfp, "%3s ", p[i].a[j].ach);
                fprintf(logfp, "%d\n", p[i].a[j].ampdef);
            }
        }
        else
            fprintf(logfp, "%s %37s\n", s, "");
    }
}

/*
 *  Title:
 *      print_defining_pha
 *  Synopsis:
 *      Prints a table with all the time-defining phases for one event
 *  Input Arguments:
 *      numphas - number of associated phases
 *      p[] - array of phase structures
 *  Calls:
 *      human_time
 *  Called by:
 *      locate_event, na_search
 */
void print_defining_pha(int numphas, PHAREC p[])
{
    int i;
    char timestr[25];
    fprintf(logfp, "    RDID      ARRID      STA                PHASE    ");
    fprintf(logfp, "TIME                     DELTA   ESAZ DELTIM RESIDUAL\n");
    for (i = 0; i < numphas; i++) {
        if (!p[i].timedef) continue;
        fprintf(logfp, "    %-9d ", p[i].rdid);
        fprintf(logfp, "%-10s ", p[i].arrid);
        fprintf(logfp,"%-18s ", p[i].fdsn);
        fprintf(logfp,"%-8s ", p[i].phase);
        human_time(timestr, p[i].time);
        fprintf(logfp,"%s ", timestr);
        fprintf(logfp,"%6.2f ", p[i].delta);
        fprintf(logfp,"%6.2f ", p[i].esaz);
        fprintf(logfp, "%6.3f ", p[i].measerr);
        fprintf(logfp, "%8.3f\n", p[i].resid);
    }
}

/*  EOF  */

