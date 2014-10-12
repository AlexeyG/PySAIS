#include <Python.h>
#include "sais.h"

#include <string.h>
#include "arrayobject.h"

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

static PyObject *python_lcp(PyObject *self, PyObject *args)
{
    PyArrayObject *SA_np, *LCP_np;
    int *SA, *LCP;
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
        LCP[k] = l;
        if (l > 0)
            l--;
    }
    free(rank);
    return Py_BuildValue("N", LCP_np);
}

static PyObject *python_lcp_int(PyObject *self, PyObject *args)
{
    PyArrayObject *T_np, *SA_np, *LCP_np;
    int *T, *SA, *LCP;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &T_np, &PyArray_Type, &SA_np))
        return NULL;
    if (SA_np == NULL)
        return NULL;
    if (not_intvector(SA_np))
        return NULL;
    if (T_np == NULL)
        return NULL;
    if (not_intvector(T_np))
        return NULL;
    SA = pyvector_to_Carrayptrs(SA_np);
    T = pyvector_to_Carrayptrs(SA_np);
    int n = SA_np->dimensions[0];
    int n_T = T_np->dimensions[0];
    if (n != strlen((const char *)T) || n != n_T)
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
        LCP[k] = l;
        if (l > 0)
            l--;
    }
    free(rank);
    return Py_BuildValue("N", LCP_np);
}
static PyMethodDef ModuleMethods[] = {
    {"sais",  python_sais, METH_VARARGS, "Construct a Suffix Array for a given string."},
    {"lcp",  python_lcp, METH_VARARGS, "Construct the corresponding LCP array given a string and its SA."},
    {"sais_int",  python_sais_int, METH_VARARGS, "Construct a Suffix Array for a given NumPy integer array."},
    {"lcp_int",  python_lcp_int, METH_VARARGS, "Construct the corresponding LCP array given a NumPy integer array and its SA."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initpysais(void)
{
    (void) Py_InitModule("pysais", ModuleMethods);
    import_array(); // NumPy
}
