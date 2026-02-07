#pragma once

#include <Python.h>
#include <torch/all.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                  \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                       \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,    \
                                        STRINGIFY(NAME),          \
                                        nullptr, 0, nullptr};    \
    return PyModule_Create(&module);                              \
  }
