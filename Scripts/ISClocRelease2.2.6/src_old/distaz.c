#include "iscloc.h"
extern int verbose;

/*
 * Functions:
 *    calc_delaz
 *    distaz
 *    deltaloc
 */

/*
 *  Title:
 *     calc_delaz
 *  Synopsis:
 *     Calculates delta, esaz and seaz for each phase.
 *  Input Arguments:
 *     sp  - pointer to current solution
 *     p[] - array of pointers to structures containing phases
 *     verbosethistime - disables verbose if 0
 *  Called by:
 *     eventloc, locate_event, fixedhypo, forward
 *  Calls:
 *     distaz, print_pha
 */
void calc_delaz(SOLREC *sp, PHAREC p[], int verbosethistime)
{
    double delta = 0., esaz = 0., seaz = 0.;
    int i;
    for (i = 0; i < sp->numphas; i++) {
        delta = distaz(p[i].sta_lat, p[i].sta_lon, sp->lat, sp->lon,
                       &seaz, &esaz, 1);
        p[i].delta = delta;
        p[i].esaz = esaz;
        p[i].seaz = seaz;
    }
    if (verbose > 2 && verbosethistime)
        print_pha(sp->numphas, p);
}

/*
 *  Title:
 *     distaz
 *  Synopsis:
 *     Computes the angular distance, azimuth and back azimuth between
 *     two geographic latitude and longitude positions
 *     (typically measured from a station to an event).
 *  Input Arguments:
 *     slat  - geographic latitude  of point 1 [deg]
 *     slon  - geographic longitude of point 1 [deg]
 *     elat  - geographic latitude  of point 2 [deg]
 *     elon  - geographic longitude of point 2 [deg]
 *     isnew - if 0, elat/elon coordinates have not changed
 *  Output Arguments:
 *     azi   - geocentric azimuth of point 2 w.r.t. point 1 (seaz)
 *     baz   - geocentric back-azimuth (esaz)
 *  Return:
 *     delta - geocentric distance between points 1 and 2 [deg]
 *  Called by:
 *     calc_delaz, distance_matrix
 */
double distaz(double slat, double slon, double elat, double elon,
              double *azi, double *baz, int isnew)
{
    double cslat = 0., sslat = 0., cdlon = 0., sdlon = 0., rdlon = 0.;
    double geoc_slat = 0., geoc_elat = 0.;
    double xazi = 0., xbaz = 0., yazi = 0., ybaz = 0., delta = 0., cdel = 0.;
    static double celat = 0., selat = 0.;
    double f = (1. - FLATTENING) * (1. - FLATTENING);
/*
 *  Convert elat from geographic latitude to geocentric latitude
 */
    if (isnew) {
        geoc_elat = atan(f * tan(DEG_TO_RAD * elat));
        celat = cos(geoc_elat);
        selat = sin(geoc_elat);
    }
    if (fabs(slat - elat) < DEPSILON && fabs(slon - elon) < DEPSILON) {
        delta = 0.0;
        *azi = 0.0;
        *baz = 180.0;
        return delta;
    }
/*
 *  Convert slat from geographic latitude to geocentric latitude
 */
    geoc_slat = atan(f * tan(DEG_TO_RAD * slat));
/*
 *  Distance
 */
    rdlon = DEG_TO_RAD * (elon - slon);
    cslat = cos(geoc_slat);
    sslat = sin(geoc_slat);
    cdlon = cos(rdlon);
    sdlon = sin(rdlon);
    cdel = sslat * selat + cslat * celat * cdlon;
    cdel = (cdel <  1.0) ? cdel :  1.0;
    cdel = (cdel > -1.0) ? cdel : -1.0;
    delta = RAD_TO_DEG * acos(cdel);
/*
 *  Azimuth, back azimuth
 */
    yazi = sdlon * celat;
    xazi = cslat * selat - sslat * celat * cdlon;
    ybaz = -sdlon * cslat;
    xbaz = celat * sslat - selat * cslat * cdlon;
    *azi = RAD_TO_DEG * atan2(yazi, xazi);
    *baz = RAD_TO_DEG * atan2(ybaz, xbaz);
    if (*azi < 0.0) *azi += 360.0;
    if (*baz < 0.0) *baz += 360.0;
    return delta;
}

/*
 *  Title:
 *     deltaloc
 *  Synopsis:
 *     Calculates geographic lat, lon of a point which is
 *         delta distance away in azim direction from a point on the Earth.
 *  Input Arguments:
 *     lat1  - geographic latitude  of point 1 [deg]
 *     lon1  - geographic longitude of point 1 [deg]
 *     delta - geocentric distance between points 1 and 2 [deg]
 *     azim  - geocentric azimuth of point 2 w.r.t. point 1
 *  Output Arguments:
 *     lat2  - geographic latitude position of point 2 [deg]
 *     lon2  - geographic longitude position of point 2 [deg]
 *  Called by:
 *     locate_event, tolatlon, get_bounce_corr
 */
void deltaloc(double lat1, double lon1, double delta, double azim,
              double *lat2, double *lon2)
{
    double a = 0., b = 0., c = 0., sinlat = 0., coslat = 0., dlon = 0.;
    double geoc_co_lat = 0.;
    double r13 = 0., r13sq = 0.;
    double x1 = 0., x2 = 0., x3 = 0., dist = 0., azi = 0.;
    double f = (1. - FLATTENING) * (1. - FLATTENING);
/*
 *  Convert a geographical location to geocentric cartesian coordinates.
 */
    dist = DEG_TO_RAD * (90.0 - delta);
    azi = DEG_TO_RAD * (180.0 - azim);
    r13 = cos(dist);
/*
 *  x1: Axis 1 intersects equator at  0 deg longitude
 *  x2: Axis 2 intersects equator at 90 deg longitude
 *  x3: Axis 3 intersects north pole
 */
    x1 = r13 * sin(azi);
    x2 = sin(dist);
    x3 = r13 * cos(azi);
    geoc_co_lat = PI2 - atan(f * tan(DEG_TO_RAD * lat1));
/*
 *  Rotate in cartesian coordinates. The cartesian coordinate system
 *  is most easily described in geocentric terms. The origin is at
 *  the Earth's center. Rotation by lat1 degrees southward, about
 *  the 1-axis.
 */
    sinlat = sin(geoc_co_lat);
    coslat = cos(geoc_co_lat);
    b = x2;
    c = x3;
    x2 = b * coslat - c * sinlat;
    x3 = b * sinlat + c * coslat;
/*
 *  Finally, convert geocentric cartesian coordinates back to
 *  a geographical location.
 */
    r13sq = x3 * x3 + x1 * x1;
    r13 = sqrt(r13sq);
    dlon = RAD_TO_DEG * atan2(x1, x3);
    a = atan2(x2, r13);
    *lat2 = RAD_TO_DEG * atan(tan(a) / f);
    *lon2 = lon1 + dlon;
    if (fabs(*lat2) > 90.0) {
        a = 180.0 - fabs(*lat2);
        *lat2 = (*lat2 >= 0.) ? -a : a;
    }
    if (fabs(*lon2) > 180.0) {
        a = 360.0 - fabs(*lon2);
        *lon2 = (*lon2 >= 0.) ? -a : a;
    }
}

/*  EOF  */
