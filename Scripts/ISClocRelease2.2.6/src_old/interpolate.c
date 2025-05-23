#include "iscloc.h"

/*
 * Functions:
 *    spline
 *    spline_int
 *    bilinear_int
 *    bracket
 *    ibracket
 */

/*
 *  Title:
 *     spline
 *  Synopsis:
 *	   Calculates interpolating coefficients for a natural spline
 *  Input Arguments:
 *     n   - number of points
 *     x   - x array
 *     y   - y array
 *  Output Arguments:
 *     d2y - second derivatives of the natural spline interpolating function
 *     tmp - temp array of n elements
 *  Called by:
 *     get_tt, read_variogram, phase_tt_h, sta_trace
 */
void spline(int n, double *x, double *y, double *d2y, double *tmp)
{
    double temp = 0., d = 0.;
    int i;
    d2y[0] = tmp[0] = 0.;
    for (i = 1; i < n - 1; i++) {
        d = (x[i] - x[i-1]) / (x[i+1] - x[i-1]);
        temp = d * d2y[i-1] + 2.;
        d2y[i] = (d - 1.) / temp;
        tmp[i] = (y[i+1] - y[i])   / (x[i+1] - x[i]) -
                 (y[i]   - y[i-1]) / (x[i]   - x[i-1]);
        tmp[i] = (6. * tmp[i] / (x[i+1] - x[i-1]) - d * tmp[i-1]) / temp;
    }
    d2y[n-1] = 0;
    for (i = n - 2; i >= 0; i--) {
        d2y[i] = d2y[i] * d2y[i+1] + tmp[i];
    }
}

/*
 *  Title:
 *     spline_int
 *  Synopsis:
 *	   Returns interpolated function value f(xp) by cubic spline interpolation
 *  Input Arguments:
 *     xp  - x point to be interpolated
 *     n   - number of points
 *     x   - x array
 *     y   - y array
 *     d2y - second derivatives of the natural spline interpolating function
 *     isderiv - calculate derivatives [0/1]
 *  Output Arguments:
 *     dydx  - first derivative
 *     d2ydx - second derivative
 *  Return:
 *     interpolated function value yp = f(xp)
 *  Called by:
 *     get_tt, data_covariance_matrix, phase_tt_h, sta_trace
 *  Calls:
 *     bracket
 */
double spline_int(double xp, int n, double *x, double *y, double *d2y,
                  int isderiv, double *dydx, double *d2ydx)
{
    double h = 0., g = 0., a = 0., b = 0., c = 0., d = 0., yp = 0.;
    int klo = 0, khi = 0;
    *dydx = *d2ydx = -999.;
/*
 *  bracket xp
 */
    bracket(xp, n, x, &klo, &khi);
/*
 *  interpolate yp
 */
    h = x[khi] - x[klo];
    g = y[khi] - y[klo];
    a = (x[khi] - xp) / h;
    b = (xp - x[klo]) / h;
    c = (a * a * a - a) * h * h / 6.;
    d = (b * b * b - b) * h * h / 6.;
    yp = a * y[klo] + b * y[khi] + c * d2y[klo] + d * d2y[khi];
/*
 *  derivatives
 */
    if (isderiv) {
        *dydx = g / h - (3. * a * a - 1.) * h * d2y[klo] / 6. +
                        (3. * b * b - 1.) * h * d2y[khi] / 6.;
        *d2ydx = a * d2y[klo] + b * d2y[khi];
    }
    return yp;
}

/*
 *  Title:
 *     bracket
 *  Synopsis:
 *	   For a vector x, ordered in ascending order, return indices jlo and jhi
 *     such that x[jlo] <= xp < x[jhi]
 *  Input Arguments:
 *     xp  - x point to be bracketed
 *     n   - number of points in x
 *     x   - x array
 *  Output Arguments:
 *     jlo - lower index
 *     jhi - upper index
 *  Called by:
 *     spline_int, bilinear_int, get_tt, phase_tt_h, sta_trace
 */
void bracket(double xp, int n, double *x, int *jlo, int *jhi)
{
    int klo = 0, khi = 0, k = 0;
    *jlo = klo = 0;
    *jhi = khi = n - 1;
    if (n < 2) return;
    while (khi - klo > 1) {
        k = (khi + klo) >> 1;
        if (x[k] > xp)
            khi = k;
        else
            klo = k;
    }
    if (klo < 0)     klo = 0;
    if (khi > n - 1) khi = n - 1;
    *jlo = klo;
    *jhi = khi;
}

/*
 *  Title:
 *     ibracket
 *  Synopsis:
 *	   For a vector x, ordered in ascending order, return indices jlo and jhi
 *     such that x[jlo] <= xp < x[jhi]
 *  Input Arguments:
 *     xp  - x point to be bracketed
 *     n   - number of points in x
 *     x   - x array
 *  Output Arguments:
 *     jlo - lower index
 *     jhi - upper index
 *  Called by:
 *     gregnum, gregtosreg
 */
void ibracket(int xp, int n, int *x, int *jlo, int *jhi)
{
    int klo = 0, khi = 0, k = 0;
    *jlo = klo = 0;
    *jhi = khi = n - 1;
    if (n < 2) return;
    while (khi - klo > 1) {
        k = (khi + klo) >> 1;
        if (x[k] > xp)
            khi = k;
        else
            klo = k;
    }
    if (klo < 0)     klo = 0;
    if (khi > n - 1) khi = n - 1;
    *jlo = klo;
    *jhi = khi;
}

/*
 *  Title:
 *     bilinear_int
 *  Synopsis:
 *	   Returns interpolated function value f(xp1,xp2) by bilinear interpolation
 *  Input Arguments:
 *     xp1  - x1 point to be interpolated
 *     xp2  - x2 point to be interpolated
 *     nx1  - number of points in x1
 *     nx2  - number of points in x2
 *     x1   - x1 vector
 *     x2   - x2 vector
 *     y    - y matrix over x1 and x2
 *  Return:
 *     interpolated function value yp = f(xp1, xp2)
 *  Called by:
 *     get_ellip_corr, get_magQ
 *  Calls:
 *     bracket
 */
double bilinear_int(double xp1, double xp2, int nx1, int nx2,
                    double *x1, double *x2, double **y)
{
    int ilo = 0, ihi = 0, jlo = 0, jhi = 0;
    double f1 = 0., f2 = 0., yp = 0.;
/*
 *  bracket xp1 and xp2
 */
    bracket(xp1, nx1, x1, &ilo, &ihi);
    bracket(xp2, nx2, x2, &jlo, &jhi);
/*
 *  scalers
 */
    f1 = (xp1 - x1[ilo]) / (x1[ihi] - x1[ilo]);
    f2 = (xp2 - x2[jlo]) / (x2[jhi] - x2[jlo]);
/*
 *  interpolate
 */
    yp = (1. - f1) * (1. - f2) * y[ilo][jlo] + f1 * (1. - f2) * y[ihi][jlo] +
         f1 * f2  * y[ihi][jhi] + (1. - f1) * f2 * y[ilo][jhi];
    return yp;
}

