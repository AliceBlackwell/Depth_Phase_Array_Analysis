#include "iscloc.h"

/*
 * Functions:
 *    epoch_time
 *    human_time
 *    epoch_time_isf
 *    human_time_isf
 *    secs
 *    read_time
 */

/*
 *  Title:
 *     epoch_time
 *  Synopsis:
 *     Converts human time to epoch time.
 *  Input Arguments:
 *     htime - human time string 'YYYY-MM-DD HH:MI:SS'
 *     msec  - fractional seconds
 *  Return:
 *     epoch time
 *  Called by:
 *     read_isf
 */
double epoch_time(char *htime, int msec)
{
    struct tm *ht = (struct tm *)NULL;
    char cutstr[25], *remnant;
    time_t et;
    int yyyy = 0, mm = 0, dd = 0, hh = 0, mi = 0, ss = 0;
    double t = 0.;
    ht = (struct tm *)calloc(1, sizeof(struct tm));
/*
 *  break down human time string
 */
    strcpy(cutstr, htime);

    cutstr[4] = '\0';
    yyyy = (int)strtod(&cutstr[0], &remnant);
    if (remnant[0]) return 0;

    cutstr[7] = '\0';
    mm = (int)strtod(&cutstr[5], &remnant);
    if (remnant[0]) return 0;

    cutstr[10] = '\0';
    dd = (int)strtod(&cutstr[8], &remnant);
    if (remnant[0]) return 0;

    cutstr[13] = '\0';
    hh = (int)strtod(&cutstr[11], &remnant);
    if (remnant[0]) return 0;

    cutstr[16] = '\0';
    mi = (int)strtod(&cutstr[14], &remnant);
    if (remnant[0]) return 0;

    cutstr[19] = '\0';
    ss = (int)strtod(&cutstr[17], &remnant);
    if (remnant[0]) return 0;
/*
 *  set time structure
 */
    ht->tm_year = yyyy - 1900;
    ht->tm_mon = mm - 1;
    ht->tm_mday = dd;
    ht->tm_hour = hh;
    ht->tm_min = mi;
    ht->tm_sec = ss;
    ht->tm_isdst = 0;
/*
 *  convert to epoch time
 */
    et = mktime(ht);
    t = (double)et + (double)msec / 1000.;
    Free(ht);
    return t;
}

/*
 *  Title:
 *     epoch_time_isf
 *  Synopsis:
 *     Converts human time to epoch time.
 *  Input Arguments:
 *     yyyy, mm, dd, hh, mi, ss, msec
 *  Return:
 *     epoch time
 *  Called by:
 *     read_isf
 */
double epoch_time_isf(int yyyy, int mm, int dd, int hh, int mi, int ss,
                      int msec)
{
    struct tm *ht = (struct tm *)NULL;
    time_t et;
    double t = 0.;
    ht = (struct tm *)calloc(1, sizeof(struct tm));
/*
 *  set time structure
 */
    ht->tm_year = yyyy - 1900;
    ht->tm_mon = mm - 1;
    ht->tm_mday = dd;
    ht->tm_hour = hh;
    ht->tm_min = mi;
    ht->tm_sec = ss;
    ht->tm_isdst = 0;
/*
 *  convert to epoch time
 */
    et = mktime(ht);
    t = (double)et + (double)msec / 1000.;
    Free(ht);
    return t;
}

/*
 *  Title:
 *     human_time
 *  Synopsis:
 *     Converts epoch time to human time.
 *  Input Arguments:
 *     etime - epoch time (double, including fractional seconds)
 *  Output Arguments:
 *     htime - human time string 'YYYY-MM-DD HH:MI:SS'
 *  Called by:
 *     eventloc, print_sol, print_hyp, print_pha, print_defining_pha,
 *     na_search, writemodels, write_isf
 */
void human_time(char *htime, double etime)
{
    struct tm *ht = (struct tm *)NULL;
    time_t et;
    int yyyy = 0, mm = 0, dd = 0, hh = 0, mi = 0, ss = 0;
    double t = 0., ft = 0., sec = 0.;
    int it = 0;
    char s[25];
    if (etime != NULLVAL) {
/*
 *      break down epoch time to day and msec used by ISC DB schema
 *      also take care of negative epoch times
 */
        sprintf(s, "%.3f", etime);
        t = atof(s);
        it = (int)t;
        sprintf(s, "%.3f\n", t - it);
        ft = atof(s);
        if (ft < 0.) {
            ft += 1.;
            t -= 1.;
        }
/*
 *      set time structure
 */
        ht = (struct tm *)calloc(1, sizeof(struct tm));
        et = (time_t)t;
        gmtime_r(&et, ht);
        yyyy = ht->tm_year + 1900;
        mm = ht->tm_mon + 1;
        dd = ht->tm_mday;
        hh = ht->tm_hour;
        mi = ht->tm_min;
        ss = ht->tm_sec;
        sec = (double)ht->tm_sec + ft;
/*
 *      human time string 'YYYY-MM-DD HH:MI:SS.SSS'
 */
        sprintf(htime, "%04d-%02d-%02d %02d:%02d:%06.3f",
                yyyy, mm, dd, hh, mi, sec);
        Free(ht);
    }
    else {
        strcpy(htime, "                       ");
    }
}

/*
 *  Title:
 *     human_time_isf
 *  Synopsis:
 *     Converts epoch time to human time.
 *  Input Arguments:
 *     etime - epoch time (double, including fractional seconds)
 *  Output Arguments:
 *     hday - human time string YYYY/MM/DD
 *     htim - human time string HH:MM:SS.SSS
 *  Called by:
 *     write_isf
 */
void human_time_isf(char *hday, char *htim, double etime)
{
    struct tm *ht = (struct tm *)NULL;
    time_t et;
    int yyyy = 0, mm = 0, dd = 0, hh = 0, mi = 0, ss = 0;
    double t = 0., ft = 0., sec = 0.;
    int it = 0;
    char s[25];
    if (etime != NULLVAL) {
/*
 *      break down epoch time to day and msec used by ISC DB schema
 *      also take care of negative epoch times
 */
        sprintf(s, "%.3f", etime);
        t = atof(s);
        it = (int)t;
        sprintf(s, "%.3f\n", t - it);
        ft = atof(s);
        if (ft < 0.) {
            ft += 1.;
            t -= 1.;
        }
/*
 *      set time structure
 */
        ht = (struct tm *)calloc(1, sizeof(struct tm));
        et = (time_t)t;
        gmtime_r(&et, ht);
        yyyy = ht->tm_year + 1900;
        mm = ht->tm_mon + 1;
        dd = ht->tm_mday;
        hh = ht->tm_hour;
        mi = ht->tm_min;
        ss = ht->tm_sec;
        sec = (double)ht->tm_sec + ft;
/*
 *      human time string 'YYYY/MM/DD HH:MI:SS.SSS'
 */
        sprintf(hday, "%04d/%02d/%02d", yyyy, mm, dd);
        sprintf(htim, "%02d:%02d:%06.3f", hh, mi, sec);
        Free(ht);
    }
    else {
        strcpy(hday, "          ");
        strcpy(htim, "            ");
    }
}

/*
 *  Title:
 *     secs
 *  Synopsis:
 *     Returns elapsed time since t0 in seconds.
 *  Input Arguments:
 *     t0 - timeval structure
 *  Return:
 *     number of seconds.
 */
double secs(struct timeval *t0)
{
    struct timeval t1;
    gettimeofday(&t1, NULL);
    return t1.tv_sec - t0->tv_sec + (double)(t1.tv_usec - t0->tv_usec) / 1000000.;
}


/*
 *  Title:
 *     read_time
 *  Synopsis:
 *     Converts time from database format string to seconds since epoch.
 *  Input Arguments:
 *     date/time as a string 'yyyy-mm-dd_hh:mi:ss.sss'
 *  Return:
 *     time - time since epoch as a double or 0 on error.
 *  Called by:
 *     read_instruction
 *  Calls:
 *     epoch_time
 */
double read_time(char *timestr)
{
    int yyyy = 0, mm = 0, dd = 0, hh = 0, mi = 0, ss = 0, msec = 0;
    char cutstr[24];
    char *remnant;

    strcpy(cutstr, timestr);

    cutstr[4] = '\0';
    yyyy = (int)strtod(&cutstr[0], &remnant);
    if (remnant[0]) return 0;

    cutstr[7] = '\0';
    mm = (int)strtod(&cutstr[5], &remnant);
    if (remnant[0]) return 0;

    cutstr[10] = '\0';
    dd = (int)strtod(&cutstr[8], &remnant);
    if (remnant[0]) return 0;

    cutstr[13] = '\0';
    hh = (int)strtod(&cutstr[11], &remnant);
    if (remnant[0]) return 0;

    cutstr[16] = '\0';
    mi = (int)strtod(&cutstr[14], &remnant);
    if (remnant[0]) return 0;

    cutstr[19] = '\0';
    ss = (int)strtod(&cutstr[17], &remnant);
    if (remnant[0]) return 0;

    if (strlen(timestr) > 20) {
        cutstr[23] = '\0';
        msec = (int)strtod(&cutstr[20], &remnant);
        if (remnant[0]) return 0;
    }
    return epoch_time_isf(yyyy, mm, dd, hh, mi, ss, msec);
}

