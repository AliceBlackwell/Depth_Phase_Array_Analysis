#include "iscloc.h"
extern int verbose;
extern FILE *logfp;
extern FILE *errfp;

#ifdef WITH_DB
extern PGconn *conn;

/*
 * Print SQL error message
 */
void pgsql_error(char *message)
{
    fprintf(errfp, "ERROR: %s %s\n", message, PQerrorMessage(conn));
    fprintf(logfp, "ERROR: %s %s\n", message, PQerrorMessage(conn));
}

/*
 * Connect to Postgres DB
 */
int pgsql_conn(void)
{
    const char *conninfo;
    conninfo = "";
    PGresult *res_set = (PGresult *)NULL;
    char sql[1536], errmsg[1536];
    conn = PQconnectdb(conninfo);
    if (PQstatus(conn) != CONNECTION_OK) {
        pgsql_error(PQerrorMessage(conn));
        pgsql_disconn();
        return 1;
    }
    sprintf(sql, "SET enable_seqscan = off");
    if ((res_set = PQexec(conn,sql)) == NULL) {
        pgsql_error("pgsql_funcs: set:");
    } else if (PQresultStatus(res_set) != PGRES_COMMAND_OK) {
        sprintf(errmsg, "pgsql_conn: %d", PQresultStatus(res_set));
        pgsql_error(errmsg);
    }
    PQclear(res_set);
    if (verbose) fprintf(stderr, "    Connected to db\n");
    return 0;
}

/*
 * Disconnect from Postgres DB
 */
void pgsql_disconn(void)
{
    PQfinish(conn);
    if (verbose) fprintf(stderr, "    Disconnected from db\n");
}


#endif
/*  EOF */
