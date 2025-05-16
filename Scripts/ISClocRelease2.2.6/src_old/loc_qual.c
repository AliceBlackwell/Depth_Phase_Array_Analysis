#include "iscloc.h"
extern int verbose;
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;

/*
 *  Title:
 *     loc_qual
 *  Synopsis:
 *     Calculates network geometry based location quality metrics
 *     gap, sgap and dU for local, near-regional, teleseismic
 *     distance ranges and the entire network.
 *         Local network:  0 - 150 km
 *         Near regional:  3 - 10 degrees
 *         Teleseismic:   28 - 180 degrees
 *         Entire network: 0 - 180 degrees
 *     Only defining stations are considered.
 *     dU is defined in:
 *        Bondár, I. and K. McLaughlin, 2009,
 *        A new ground truth data set for seismic studies,
 *        Seism. Res. Let., 80, 465-472.
 *     sgap is defined in:
 *        Bondár, I., S.C. Myers, E.R. Engdahl and E.A. Bergman, 2004,
 *        Epicenter accuracy based on seismic network criteria,
 *        Geophys. J. Int., 156, 483-496, doi: 10.1111/j.1365-246X.2004.02070.x.
 *  Input Arguments:
 *     hypid   - hypocentre id
 *     numphas - number of associated phases
 *     p[]     - array of phase structures
 *  Output Arguments:
 *     hq      - pointer to hypocentre quality structure
 *  Return:
 *     0/1 on success/error
 *  Called by:
 *     eventloc
 *  Calls:
 *     gapper
 */
int location_quality(int numphas, PHAREC p[], HYPQUAL *hq)
{
    double *esaz = (double *)NULL;
    double gap = 0., sgap = 0., du = 0., score = 0.;
    double delta = 0., d10 = 0., mind = 0., maxd = 0., d = 0.;
    char prevsta[STALEN];
    int i, ndef = 0, nsta = 0, ndefsta_10km = 0;
    if ((esaz = (double *)calloc(numphas + 2, sizeof(double))) == NULL) {
        fprintf(logfp, "location_quality: cannot allocate memory\n");
        fprintf(errfp, "location_quality: cannot allocate memory\n");
        errorcode = 1;
        return 1;
    }
/*
 *  local network (0-150 km)
 */
    delta = 150. * RAD_TO_DEG / EARTH_RADIUS;
    d10 = 10. * RAD_TO_DEG / EARTH_RADIUS;
    strcpy(prevsta, "");
    mind = 180.;
    maxd = 0.;
    for (ndef = nsta = 0, i = 0; i < numphas; i++) {
        if (!p[i].timedef) continue;
        if (p[i].delta > delta) continue;
        ndef++;
        if (streq(p[i].prista, prevsta)) continue;
        esaz[nsta++] = p[i].esaz;
        strcpy(prevsta, p[i].prista);
        if (p[i].delta > maxd) maxd = p[i].delta;
        if (p[i].delta < mind) mind = p[i].delta;
        if (p[i].delta <= d10) ndefsta_10km++;
    }
    hq->local_net.ndefsta = nsta;
    hq->local_net.ndef = ndef;
    du = gapper(nsta, esaz, &gap, &sgap);
    hq->local_net.du = du;
    hq->local_net.gap = gap;
    hq->local_net.sgap = sgap;
    hq->local_net.mindist = mind;
    hq->local_net.maxdist = maxd;
    hq->ndefsta_10km = ndefsta_10km;
    hq->gtcand = (du > 0.35 || ndefsta_10km < 1 || sgap > 160.) ? 0 : 1;
    d = (du < 0.0001) ? 0.0001 : du;
    score += 2. * (1. / d + nsta / 7.5 + (360. - sgap) / 60.);
    if (verbose > 1) {
        fprintf(logfp, "    local network:         nsta=%3d ndef=%3d",
                nsta, ndef);
        fprintf(logfp, " gap=%5.1f sgap=%5.1f dU=%5.3f", gap, sgap, du);
        fprintf(logfp, " ndefsta_10km=%d GT5cand=%d\n",
                ndefsta_10km, hq->gtcand);
    }
/*
 *  near-regional network (3-10 degrees)
 */
    strcpy(prevsta, "");
    mind = 180.;
    maxd = 0.;
    for (ndef = nsta = 0, i = 0; i < numphas; i++) {
        if (!p[i].timedef) continue;
        if (p[i].delta < 3. || p[i].delta > 10.) continue;
        ndef++;
        if (streq(p[i].prista, prevsta)) continue;
        esaz[nsta++] = p[i].esaz;
        strcpy(prevsta, p[i].prista);
        if (p[i].delta > maxd) maxd = p[i].delta;
        if (p[i].delta < mind) mind = p[i].delta;
    }
    hq->near_net.ndefsta = nsta;
    hq->near_net.ndef = ndef;
    du = gapper(nsta, esaz, &gap, &sgap);
    hq->near_net.du = du;
    hq->near_net.gap = gap;
    hq->near_net.sgap = sgap;
    hq->near_net.mindist = mind;
    hq->near_net.maxdist = maxd;
    d = (du < 0.0001) ? 0.0001 : du;
    score += 1.2 * (1. / d + nsta / 7.5 + (360. - sgap) / 60.);
    if (verbose > 1) {
        fprintf(logfp, "    near-regional network: nsta=%3d ndef=%3d",
                nsta, ndef);
        fprintf(logfp, " gap=%5.1f sgap=%5.1f dU=%5.3f\n", gap, sgap, du);
    }
/*
 *  teleseismic network (28-180 degrees)
 */
    strcpy(prevsta, "");
    mind = 180.;
    maxd = 0.;
    for (ndef = nsta = 0, i = 0; i < numphas; i++) {
        if (!p[i].timedef) continue;
        if (p[i].delta < 28.) continue;
        ndef++;
        if (streq(p[i].prista, prevsta)) continue;
        esaz[nsta++] = p[i].esaz;
        strcpy(prevsta, p[i].prista);
        if (p[i].delta > maxd) maxd = p[i].delta;
        if (p[i].delta < mind) mind = p[i].delta;
    }
    hq->tele_net.ndefsta = nsta;
    hq->tele_net.ndef = ndef;
    du = gapper(nsta, esaz, &gap, &sgap);
    hq->tele_net.du = du;
    hq->tele_net.gap = gap;
    hq->tele_net.sgap = sgap;
    hq->tele_net.mindist = mind;
    hq->tele_net.maxdist = maxd;
    d = (du < 0.0001) ? 0.0001 : du;
    score += 1.5 * (1. / d + nsta / 7.5 + (360. - sgap) / 60.);
    if (verbose > 1) {
        fprintf(logfp, "    teleseismic network:   nsta=%3d ndef=%3d",
                nsta, ndef);
        fprintf(logfp, " gap=%5.1f sgap=%5.1f dU=%5.3f\n", gap, sgap, du);
    }
/*
 *  entire network
 */
    strcpy(prevsta, "");
    mind = 180.;
    maxd = 0.;
    for (ndef = nsta = 0, i = 0; i < numphas; i++) {
        if (!p[i].timedef) continue;
        ndef++;
        if (streq(p[i].prista, prevsta)) continue;
        esaz[nsta++] = p[i].esaz;
        strcpy(prevsta, p[i].prista);
        if (p[i].delta > maxd) maxd = p[i].delta;
        if (p[i].delta < mind) mind = p[i].delta;
    }
    hq->whole_net.ndefsta = nsta;
    hq->whole_net.ndef = ndef;
    du = gapper(nsta, esaz, &gap, &sgap);
    hq->whole_net.du = du;
    hq->whole_net.gap = gap;
    hq->whole_net.sgap = sgap;
    hq->whole_net.mindist = mind;
    hq->whole_net.maxdist = maxd;
    d = (du < 0.0001) ? 0.0001 : du;
    score += (1. / d + nsta / 7.5 + (360. - sgap) / 60.);
    if (verbose > 1) {
        fprintf(logfp, "    entire network:        nsta=%3d ndef=%3d",
                nsta, ndef);
        fprintf(logfp, " gap=%5.1f sgap=%5.1f dU=%5.3f\n", gap, sgap, du);
    }
    if (score < 0.1) score = 4;
    hq->score = score;
    Free(esaz);
    return 0;
}

/*
 *  Title:
 *     gapper
 *  Synopsis:
 *     Calculates gap, sgap and dU.
 *
 *           4 * sum|esaz[i] - (360 *i / nsta + b)|
 *     dU = ---------------------------------------
 *                      360 * nsta
 *
 *     b = avg(esaz) - avg(360i/N)  where esaz is sorted
 *
 *  Input Arguments:
 *     nsta - number of defining stations
 *     esaz - array of event-to-station azimuths
 *  Output Arguments:
 *     gap  - largest azimuthal gap
 *     sgap - largest secondary azimuthal gap
 *  Return:
 *     dU   - network quality metric
 *  Called by:
 *     location_quality
 */
double gapper(int nsta, double *esaz, double *gap, double *sgap)
{
    int i;
    double du = 1., bb = 0., uesaz = 0., w = 0., s1 = 0., s2 = 0.;
    *gap = 360.;
    *sgap = 360.;
    if (nsta < 2) return du;
/*
 *  sort esaz
 */
    qsort(esaz, nsta, sizeof(double), double_compare);
/*
 *  du: mean absolute deviation from best fitting uniform network
 */
    for (i = 0; i < nsta; i++) {
        uesaz = 360. * (double)i / (double)nsta;
        s1 += esaz[i];
        s2 += uesaz;
    }
    bb = (s1 - s2) / (double)nsta;
    for (w = 0., i = 0; i < nsta; i++) {
        uesaz = 360. * (double)i / (double)nsta;
        w += fabs(esaz[i] - uesaz - bb);
    }
    du = 4. * w / (360. * (double)nsta);
/*
 *  gap
 */
    esaz[nsta] = esaz[0] + 360.;
    for (w = 0., i = 0; i < nsta; i++)
        w = max(w, esaz[i+1] - esaz[i]);
    if (w > 360.) w = 360.;
    *gap = w;
/*
 *  sgap
 */
    esaz[nsta+1] = esaz[1] + 360.;
    for (w = 0., i = 0; i < nsta; i++)
        w = max(w, esaz[i+2] - esaz[i]);
    if (w > 360.) w = 360.;
    *sgap = w;
    return du;
}


/*  EOF  */
