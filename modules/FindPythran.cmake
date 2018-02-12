# - Find the Pythran's Pythonic libraries
# This module finds if Pythran is installed, and sets the following variables
# indicating where it is.
#
#  Pythran_FOUND              - was Pythran found
#  Pythran_VERSION            - the version of Pythran found as a string
#  Pythran_VERSION_MAJOR      - the major version number of Pythran
#  Pythran_VERSION_MINOR      - the minor version number of Pythran
#  Pythran_VERSION_PATCH      - the patch version number of Pythran
#  Pythran_VERSION_DECIMAL    - e.g. version 1.6.1 is 10601
#  Pythran_INCLUDE_DIRS       - path to the Pythran include files

# Based on FindNumpy.cmake
# (Copyright 2012 Continuum Analytics, Inc. -- MIT License)

# Finding Pythran involves calling the Python interpreter
if(Pythran_FIND_REQUIRED)
    find_package(PythonInterp REQUIRED)
else()
    find_package(PythonInterp)
endif()

if(NOT PYTHONINTERP_FOUND)
    set(Pythran_FOUND FALSE)
endif()

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import logging; logging.getLogger('pythran').disabled = True; import inspect; import os; import pythran; print(pythran.__version__); print(os.path.dirname(inspect.getfile(pythran)));"
    RESULT_VARIABLE _Pythran_SEARCH_SUCCESS
    OUTPUT_VARIABLE _Pythran_VALUES
    ERROR_VARIABLE _Pythran_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT _Pythran_SEARCH_SUCCESS MATCHES 0)
    if(Pythran_FIND_REQUIRED)
        message(FATAL_ERROR
            "Pythran import failure:\n${_Pythran_ERROR_VALUE}")
    endif()
    set(Pythran_FOUND FALSE)
endif()

# Convert the process output into a list
string(REGEX REPLACE ";" "\\\\;" _Pythran_VALUES ${_Pythran_VALUES})
string(REGEX REPLACE "\n" ";" _Pythran_VALUES ${_Pythran_VALUES})
list(GET _Pythran_VALUES 0 Pythran_VERSION)
list(GET _Pythran_VALUES 1 Pythran_INCLUDE_DIRS)

# Make sure all directory separators are '/'
string(REGEX REPLACE "\\\\" "/" Pythran_INCLUDE_DIRS ${Pythran_INCLUDE_DIRS})

# Get the major and minor version numbers
string(REGEX REPLACE "\\." ";" _Pythran_VERSION_LIST ${Pythran_VERSION})
list(GET _Pythran_VERSION_LIST 0 Pythran_VERSION_MAJOR)
list(GET _Pythran_VERSION_LIST 1 Pythran_VERSION_MINOR)
list(GET _Pythran_VERSION_LIST 2 Pythran_VERSION_PATCH)
string(REGEX MATCH "[0-9]*" Pythran_VERSION_PATCH ${Pythran_VERSION_PATCH})
math(EXPR Pythran_VERSION_DECIMAL
    "(${Pythran_VERSION_MAJOR} * 10000) + (${Pythran_VERSION_MINOR} * 100) + ${Pythran_VERSION_PATCH}")

find_package_message(Pythran
    "Found Pythran (Pythonic): version \"${Pythran_VERSION}\" ${Pythran_INCLUDE_DIRS}"
    "${Pythran_INCLUDE_DIRS}${Pythran_VERSION}")

set(Pythran_FOUND TRUE)
