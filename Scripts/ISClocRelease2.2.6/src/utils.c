#include "iscloc.h"
extern int verbose;
extern FILE *logfp;
extern FILE *errfp;
extern int errorcode;

/*
 *  Title:
 *     skipcomments
 *  Synopsis:
 *     Skips comment lines in file. A comment line starts with #.
 *  Input Arguments:
 *     buf - current line from file
 *     fp  - file pointer
 *  Output Arguments:
 *     buf - next non-comment line
 */
void skipcomments(char *buf, FILE *fp)
{
    do {
        fgets(buf, LINLEN, fp);
        rtrim(buf);                                 /* cut trailing spaces */
    } while (buf[0] == '#' || buf[0] == '\0');
}

/*
 *  Title:
 *     rtrim
 *  Synopsis:
 *     Cuts trailing spaces from string.
 *  Input Arguments:
 *     buf - input string
 *  Return:
 *     buf - input string, with trailing spaces removed
 */
char *rtrim(char *buf)
{
    int n = (int)strlen(buf) - 1;
    if (buf[n] == '\n') n--;
    while (buf[n] == ' ') n--;
    buf[n+1] = '\0';
    return buf;
}

/*
 *  Title:
 *     dropspace
 *  Synopsis:
 *     Removes extra white space characters from string.
 *     Also removes spaces around equal sign.
 *  Input Arguments:
 *     str1 - string to be de-white-spaced
 *  Output Arguments:
 *     str2 - string with white space characters removed
 *  Return:
 *     length of resulting string
 */
int dropspace(char *str1, char *str2)
{
    int i, j, n, isch = 0;
    char c;
    n = strlen(str1);
    for (j = 0, i = 0; i < n; i++) {
        c = str1[i];
        if (isblank(c)) {
            if (isch)
                str2[j++] = c;
            isch = 0;
        }
        else {
/*
 *          remove spaces around equal sign
 */ 
            if (c == '=') {
                if (i) {
                    if (isblank(str1[i-1]))
                        j--;
                }
                if (i < n - 1) {
                    if (isblank(str1[i+1]))
                        i++;
                }
            }
            str2[j++] = c;                
            isch = 1;
        }
    }
    str2[j] = '\0';
    return j;
}

/*
 *
 * Free: a smart free
 *
 */
void Free(void *ptr)
{
    if (ptr != NULL)
        free(ptr);
}

/*
 *  Title:
 *     alloc_matrix
 *  Synopsis:
 *     Allocates memory to a double matrix.
 *  Input Arguments:
 *     nrow - number of rows
 *     ncol - number of columns
 *  Returns:
 *     pointer to matrix
 */
double **alloc_matrix(int nrow, int ncol)
{
    double **matrix = (double **)NULL;
    int i;
    if ((matrix = (double **)calloc(nrow, sizeof(double *))) == NULL) {
        fprintf(logfp, "alloc_matrix: cannot allocate memory\n");
        fprintf(errfp, "alloc_matrix: cannot allocate memory\n");
        errorcode = 1;
        return (double **)NULL;
    }
    if ((matrix[0] = (double *)calloc(nrow * ncol, sizeof(double))) == NULL) {
        fprintf(logfp, "alloc_matrix: cannot allocate memory\n");
        fprintf(errfp, "alloc_matrix: cannot allocate memory\n");
        Free(matrix);
        errorcode = 1;
        return (double **)NULL;
    }
    for (i = 1; i < nrow; i++)
        matrix[i] = matrix[i - 1] + ncol;
    return matrix;
}

/*
 *  Title:
 *     free_matrix
 *  Synopsis:
 *     Frees memory allocated to a matrix.
 *  Input Arguments:
 *     matrix - matrix
 */
void free_matrix(double **matrix)
{
    if (matrix != NULL) {
        Free(matrix[0]);
        Free(matrix);
    }
}

/*
 *  Title:
 *     alloc_i2matrix
 *  Synopsis:
 *     Allocates memory to a short integer matrix.
 *  Input Arguments:
 *     nrow - number of rows
 *     ncol - number of columns
 *  Returns:
 *     pointer to matrix
 */
short int **alloc_i2matrix(int nrow, int ncol)
{
    short int **matrix = (short int **)NULL;
    int i;
    if ((matrix = (short int **)calloc(nrow, sizeof(short int *))) == NULL) {
        fprintf(logfp, "alloc_i2matrix: cannot allocate memory\n");
        fprintf(errfp, "alloc_i2matrix: cannot allocate memory\n");
        errorcode = 1;
        return (short int **)NULL;
    }
    if ((matrix[0] = (short int *)calloc(nrow * ncol, sizeof(short int))) == NULL) {
        fprintf(logfp, "alloc_i2matrix: cannot allocate memory\n");
        fprintf(errfp, "alloc_i2matrix: cannot allocate memory\n");
        Free(matrix);
        errorcode = 1;
        return (short int **)NULL;
    }
    for (i = 1; i < nrow; i++)
        matrix[i] = matrix[i - 1] + ncol;
    return matrix;
}

/*
 *  Title:
 *     free_i2matrix
 *  Synopsis:
 *     Frees memory allocated to a short integer matrix.
 *  Input Arguments:
 *     matrix - matrix
 */
void free_i2matrix(short int **matrix)
{
    if (matrix != NULL) {
        Free(matrix[0]);
        Free(matrix);
    }
}

/*
 *  Title:
 *     int_compare
 *  Synopsis:
 *     compares two ints
 *  Returns:
 *     -1 if x < y, 1 if x > y and 0 if x == y
 */
int int_compare(const void *x, const void *y)
{
    if (*(int *)x < *(int *)y)
        return -1;
    if (*(int *)x > *(int *)y)
        return 1;
    return 0;
}

/*
 *  Title:
 *     double_compare
 *  Synopsis:
 *     compares two doubles
 *  Returns:
 *     -1 if x < y, 1 if x > y and 0 if x == y
 */
int double_compare(const void *x, const void *y)
{
    if (*(double *)x < *(double *)y)
        return -1;
    if (*(double *)x > *(double *)y)
        return 1;
    return 0;
}

/*  EOF  */
