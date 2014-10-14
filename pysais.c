#include <Python.h>
#include "sais.h"

#include <string.h>
#include "arrayobject.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

/* C vector utility functions */ 
PyArrayObject *pyvector(PyObject *objin);
int *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
int  not_intvector(PyArrayObject *vec);

/* Vector Utility functions */
PyArrayObject *pyvector(PyObject *objin) 
{
    return (PyArrayObject *) PyArray_ContiguousFromObject(objin, NPY_INT, 1, 1); 
}

/* Create 1D Carray from PyArray */
int *pyvector_to_Carrayptrs(PyArrayObject *arrayin) 
{
    return (int *) arrayin->data;  /* pointer to arrayin data as double */
}

/* Check that PyArrayObject is an int type and a vector */ 
int  not_intvector(PyArrayObject *vec)
{
    if (vec->descr->type_num != NPY_INT || vec->nd != 1)
    {  
        PyErr_SetString(PyExc_ValueError, "Array must be of type Int and 1 dimensional (n).");
        return 1;
    }
    return 0;
}

static PyObject *python_sais(PyObject *self, PyObject *args)
{
    const unsigned char *T;
    PyArrayObject *SA_np;
    int *SA;
    if (!PyArg_ParseTuple(args, "s", &T))
        return NULL;
    int n = strlen((const char *)T);
    int dims[2];
    dims[0] = n;
    SA_np = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_INT);
    SA = pyvector_to_Carrayptrs(SA_np);
    int res = sais(T, SA, n);
    if (res < 0)
    {
        PyErr_SetString(PyExc_StopIteration, "Error occurred in SA-IS.");
        return NULL;
    }
    return Py_BuildValue("N", SA_np);
}

static PyObject *python_sais_int(PyObject *self, PyObject *args)
{
    PyArrayObject *T_np, *SA_np;
    int *T, *SA;
    int i, k;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &T_np, &k))
        return NULL;
    if (T_np == NULL)
        return NULL;
    if (not_intvector(T_np))
        return NULL;
    if (k <= 0)
        return NULL;
    T = pyvector_to_Carrayptrs(T_np);
    int n = T_np->dimensions[0];
    for (i = 0; i < n; i++)
        if (T[i] < 0 || T[i] >= k)
        {
            PyErr_SetString(PyExc_StopIteration, "Array elements must be >= 0 and < k (alphabet size).");
            return NULL;
        }
    int dims[2];
    dims[0] = n;
    SA_np = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_INT);
    SA = pyvector_to_Carrayptrs(SA_np);
    int res = sais_int(T, SA, n, k);
    if (res < 0)
    {
        PyErr_SetString(PyExc_StopIteration, "Error occurred in SA-IS.");
        return NULL;
    }
    return Py_BuildValue("N", SA_np);
}

static int __lcp_left_right(int *LCP, int *LCP_left, int *LCP_right, int left, int right)
{
    if (left == right - 1)
        return LCP[left];
    int middle = (left + right) / 2;
    LCP_left[middle - 1] = __lcp_left_right(LCP, LCP_left, LCP_right, left, middle);
    LCP_right[middle - 1] = __lcp_left_right(LCP, LCP_left, LCP_right, middle, right);
    //printf("(%d, %d, %d) : %d\n", left, middle, right, MIN(LCP_left[middle - 1], LCP_right[middle - 1]));
    return MIN(LCP_left[middle - 1], LCP_right[middle - 1]);
}

static int __bisect_sa(const unsigned char *T, const unsigned char *P, int *SA, int *LCP_left, int *LCP_right, int n, int m, char *found)
{
    int left = 0, right = n - 1;
    int lcp_l = 0, lcp_r = 0;
    int offset = SA[n - 1];
    while (lcp_r < m && offset + lcp_r < n && P[lcp_r] == T[offset + lcp_r]) lcp_r++;
    int i = 0;
    int middle, lcp_mr, lcp_lm;
    char is_pattern_less;
    char solved;
    while (1)
    {
        middle = (left + right) / 2;
        //printf("(%d, %d, %d)\n", left, middle, right);
        is_pattern_less = 1;
        solved = 0;
        i = MAX(lcp_l, lcp_r);
        if (lcp_l > lcp_r)
        {
            lcp_lm = LCP_left[middle - 1];
            if (lcp_lm > i)
            {
                is_pattern_less = 0;
                solved = 1;
            }
            else if (lcp_lm < i)
            {
                lcp_r = lcp_lm;
                solved = 1;
            }
        }
        else if (lcp_l < lcp_r)
        {
            lcp_mr = LCP_right[middle - 1];
            if (lcp_mr > i)
            {
                solved = 1;
            }
            else if (lcp_mr < i)
            {
                is_pattern_less = 0;
                lcp_l = lcp_mr;
                solved = 1;
            }
        }
        
        if (!solved)
        {
            offset = SA[middle];
            while (i < m && offset + i < n)
            {
                if (P[i] < T[offset + i])
                    break;
                if (P[i] > T[offset + i])
                {
                    is_pattern_less = 0;
                    break;
                }
                i++;
            }
        }
        //printf("is_pattern_less = %d, i = %d, solved = %d \n", is_pattern_less, i, solved);

        if (is_pattern_less)
        {
            if (middle == left + 1)
            {
                *found = i == m;
                return middle;
            }
            right = middle;
            if (!solved)
                lcp_r = i;
        }
        else
        {
            if (middle == right - 1)
            {
                *found = i == m;
                return right;
            }
            left = middle;
            if (!solved)
                lcp_l = i;
        }
    }
}

static PyObject *python_lcp(PyObject *self, PyObject *args)
{
    PyArrayObject *SA_np, *LCP_np, *LCP_left_np, *LCP_right_np;
    int *SA, *LCP, *LCP_left, *LCP_right;
    const unsigned char *T;
    if (!PyArg_ParseTuple(args, "sO!", &T, &PyArray_Type, &SA_np))
        return NULL;
    if (SA_np == NULL)
        return NULL;
    if (not_intvector(SA_np))
        return NULL;
    SA = pyvector_to_Carrayptrs(SA_np);
    int n = SA_np->dimensions[0];
    if (n != strlen((const char *)T))
        return NULL;
    int i;
    for (i = 0; i < n; i++)
        if (SA[i] < 0 || SA[i] >= n)
        {
            PyErr_SetString(PyExc_StopIteration, "Incorrect SA given as input.");
            return NULL;
        }
    int dims[2];
    dims[0] = n;
    LCP_np = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_INT);
    LCP = pyvector_to_Carrayptrs(LCP_np);
    dims[0]--;
    LCP_left_np = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_INT);
    LCP_left = pyvector_to_Carrayptrs(LCP_left_np);
    LCP_right_np = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_INT);
    LCP_right = pyvector_to_Carrayptrs(LCP_right_np);
    int *rank = malloc(n * sizeof(int));
    if (rank == NULL)
    {
        PyErr_SetString(PyExc_StopIteration, "Unable to allocate memory.");
        return NULL;
    }
    int l, j, k;
    for (i = 0; i < n; i++)
        rank[SA[i]] = i;
    l = 0;
    for (i = 0; i < n; i++)
    {
        k = rank[i];
        j = SA[k - 1];
        while (T[i + l] == T[j + l])
            l++;
        if (k > 0)
            LCP[k - 1] = l;
        else
            LCP[n - 1] = l;
        if (l > 0)
            l--;
    }
    free(rank);
    __lcp_left_right(LCP, LCP_left, LCP_right, 0, n - 1);
    return Py_BuildValue("NNN", LCP_np, LCP_left_np, LCP_right_np);
}

static PyObject *python_lcp_int(PyObject *self, PyObject *args)
{
    PyArrayObject *T_np, *SA_np, *LCP_np, *LCP_left_np, *LCP_right_np;
    int *T, *SA, *LCP, *LCP_left, *LCP_right;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &T_np, &PyArray_Type, &SA_np))
        return NULL;
    if (SA_np == NULL)
    {
        PyErr_SetString(PyExc_StopIteration, "SA cannot be None.");
        return NULL;
    }
    if (not_intvector(SA_np))
        return NULL;
    if (T_np == NULL)
    {
        PyErr_SetString(PyExc_StopIteration, "T cannot be None.");
        return NULL;
    }
    if (not_intvector(T_np))
        return NULL;
    SA = pyvector_to_Carrayptrs(SA_np);
    T = pyvector_to_Carrayptrs(SA_np);
    int n = SA_np->dimensions[0];
    int n_T = T_np->dimensions[0];
    if (n != n_T)
    {
        PyErr_SetString(PyExc_StopIteration, "SA and NumPy array lengths do not match.");
        return NULL;
    }
    int i;
    for (i = 0; i < n; i++)
        if (SA[i] < 0 || SA[i] >= n)
        {
            PyErr_SetString(PyExc_StopIteration, "Incorrect SA given as input.");
            return NULL;
        }
    int dims[2];
    dims[0] = n;
    LCP_np = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_INT);
    LCP = pyvector_to_Carrayptrs(LCP_np);
    dims[0]--;
    LCP_left_np = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_INT);
    LCP_left = pyvector_to_Carrayptrs(LCP_left_np);
    LCP_right_np = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_INT);
    LCP_right = pyvector_to_Carrayptrs(LCP_right_np);
    int *rank = malloc(n * sizeof(int));
    if (LCP == NULL || rank == NULL)
    {
        PyErr_SetString(PyExc_StopIteration, "Unable to allocate memory.");
        return NULL;
    }
    int l, j, k;
    for (i = 0; i < n; i++)
        rank[SA[i]] = i;
    l = 0;
    for (i = 0; i < n; i++)
    {
        k = rank[i];
        j = SA[k - 1];
        while (T[i + l] == T[j + l])
            l++;
        if (k > 0)
            LCP[k - 1] = l;
        else
            LCP[n - 1] = l;
        if (l > 0)
            l--;
    }
    free(rank);
    __lcp_left_right(LCP, LCP_left, LCP_right, 0, n - 1);
    return Py_BuildValue("NNN", LCP_np, LCP_left_np, LCP_right_np);
}

PyObject *python_bisect(PyObject *self, PyObject *args)
{
    PyArrayObject *SA_np, *LCP_left_np, *LCP_right_np; 
    int *SA, *LCP_left, *LCP_right;
    const unsigned char *T, *P;
    if (!PyArg_ParseTuple(args, "ssO!O!O!", &T, &P, &PyArray_Type, &SA_np, &PyArray_Type, &LCP_left_np, &PyArray_Type, &LCP_right_np))
        return NULL;
    if (SA_np == NULL)
        return NULL;
    if (LCP_left_np == NULL)
        return NULL;
    if (LCP_right_np == NULL)
        return NULL;
    SA = pyvector_to_Carrayptrs(SA_np);
    LCP_left = pyvector_to_Carrayptrs(LCP_left_np);
    LCP_right = pyvector_to_Carrayptrs(LCP_right_np);
    int n = SA_np->dimensions[0];
    int m = strlen((const char *)P);
    char found;
    int index = __bisect_sa(T, P, SA, LCP_left, LCP_right, n, m, &found);
    return Py_BuildValue("iO", index, found ? Py_True : Py_False);
}

PyObject *python_count_occurrences(PyObject *self, PyObject *args)
{
    PyArrayObject *SA_np, *LCP_np, *LCP_left_np, *LCP_right_np, *assignment_np; 
    int *SA, *LCP, *LCP_left, *LCP_right, *assignment;
    const unsigned char *T, *P;
    int n_samples, str_length;
    if (!PyArg_ParseTuple(args, "ssO!O!O!O!O!ii", &T, &P, &PyArray_Type, &assignment_np, &PyArray_Type, &SA_np, &PyArray_Type, &LCP_np, &PyArray_Type, &LCP_left_np, &PyArray_Type, &LCP_right_np, &n_samples, &str_length))
        return NULL;
    if (assignment_np == NULL)
        return NULL;
    if (SA_np == NULL)
        return NULL;
    if (LCP_np == NULL)
        return NULL;
    if (LCP_left_np == NULL)
        return NULL;
    if (LCP_right_np == NULL)
        return NULL;
    assignment = pyvector_to_Carrayptrs(assignment_np);
    SA = pyvector_to_Carrayptrs(SA_np);
    LCP = pyvector_to_Carrayptrs(LCP_np);
    LCP_left = pyvector_to_Carrayptrs(LCP_left_np);
    LCP_right = pyvector_to_Carrayptrs(LCP_right_np);
    int n = SA_np->dimensions[0];
    int m = strlen((const char *)P);
    char found;
    int i = __bisect_sa(T, P, SA, LCP_left, LCP_right, n, m, &found);
    
    int dims[2];
    dims[0] = n_samples;
    dims[1] = 3;
    PyArrayObject *counts_np;
    char *counts;
    int j, k;
    counts_np = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_BYTE);
    if (found)
    {
        counts = (char *)counts_np->data;
        str_length++;
        while (i < n && LCP[i] >= m)
        {
            k = SA[i];
            j = (k % str_length) % 3;
            (*(counts + 3 * assignment[k] + j))++;
            i++;
        }
    }
    return Py_BuildValue("N", counts_np);
}

static PyMethodDef ModuleMethods[] = {
    {"sais",  python_sais, METH_VARARGS, "Construct a Suffix Array for a given string."},
    {"lcp",  python_lcp, METH_VARARGS, "Construct the corresponding LCP array given a string and its SA."},
    {"sais_int",  python_sais_int, METH_VARARGS, "Construct a Suffix Array for a given NumPy integer array."},
    {"lcp_int",  python_lcp_int, METH_VARARGS, "Construct the corresponding LCP array given a NumPy integer array and its SA."},
    {"bisect",  python_bisect, METH_VARARGS, "Query the SA using bisection. Inputs are Text, Pattern, SA, LCP_left, LCP_right."},
    {"count_occurrences",  python_count_occurrences, METH_VARARGS, "A quick and dirty ad-hoc occurrence counting code."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initpysais(void)
{
    (void) Py_InitModule("pysais", ModuleMethods);
    import_array(); // NumPy
}
