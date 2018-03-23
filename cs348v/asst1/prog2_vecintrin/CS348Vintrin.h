// Define vector unit width here
#define VECTOR_WIDTH 4

#ifndef CS348VINTRIN_H_
#define CS348VINTRIN_H_

#include <cstdlib>
#include <cmath>
#include "logger.h"

//*******************
//* Type Definition *
//*******************

extern Logger CS348VLogger;

template <typename T>
struct __cs348v_vec {
  T value[VECTOR_WIDTH];
};

// Declare a mask with __cs348v_mask
struct __cs348v_mask : __cs348v_vec<bool> {};

// Declare a floating point vector register with __cs348v_vec_float
#define __cs348v_vec_float __cs348v_vec<float>

// Declare an integer vector register with __cs348v_vec_int
#define __cs348v_vec_int   __cs348v_vec<int>

//***********************
//* Function Definition *
//***********************

// Return a mask initialized to 1 in the first N lanes and 0 in the others
__cs348v_mask _cs348v_init_ones(int first = VECTOR_WIDTH);

// Return the inverse of maska
__cs348v_mask _cs348v_mask_not(__cs348v_mask &maska);

// Return (maska | maskb)
__cs348v_mask _cs348v_mask_or(__cs348v_mask &maska, __cs348v_mask &maskb);

// Return (maska & maskb)
__cs348v_mask _cs348v_mask_and(__cs348v_mask &maska, __cs348v_mask &maskb);

// Count the number of 1s in maska
int _cs348v_cntbits(__cs348v_mask &maska);

// Set register to value if vector lane is active
//  otherwise keep the old value
void _cs348v_vset_float(__cs348v_vec_float &vecResult, float value, __cs348v_mask &mask);
void _cs348v_vset_int(__cs348v_vec_int &vecResult, int value, __cs348v_mask &mask);
// For user's convenience, returns a vector register with all lanes initialized to value
__cs348v_vec_float _cs348v_vset_float(float value);
__cs348v_vec_int _cs348v_vset_int(int value);

// Copy values from vector register src to vector register dest if vector lane active
// otherwise keep the old value
void _cs348v_vmove_float(__cs348v_vec_float &dest, __cs348v_vec_float &src, __cs348v_mask &mask);
void _cs348v_vmove_int(__cs348v_vec_int &dest, __cs348v_vec_int &src, __cs348v_mask &mask);

// Load values from array src to vector register dest if vector lane active
//  otherwise keep the old value
void _cs348v_vload_float(__cs348v_vec_float &dest, float* src, __cs348v_mask &mask);
void _cs348v_vload_int(__cs348v_vec_int &dest, int* src, __cs348v_mask &mask);

// Store values from vector register src to array dest if vector lane active
//  otherwise keep the old value
void _cs348v_vstore_float(float* dest, __cs348v_vec_float &src, __cs348v_mask &mask);
void _cs348v_vstore_int(int* dest, __cs348v_vec_int &src, __cs348v_mask &mask);

// Return calculation of (veca + vecb) if vector lane active
//  otherwise keep the old value
void _cs348v_vadd_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
void _cs348v_vadd_int(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

// Return calculation of (veca - vecb) if vector lane active
//  otherwise keep the old value
void _cs348v_vsub_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
void _cs348v_vsub_int(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

// Return calculation of (veca * vecb) if vector lane active
//  otherwise keep the old value
void _cs348v_vmult_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
void _cs348v_vmult_int(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

// Return calculation of (veca / vecb) if vector lane active
//  otherwise keep the old value
void _cs348v_vdiv_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
void _cs348v_vdiv_int(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);


// Return calculation of absolute value abs(veca) if vector lane active
//  otherwise keep the old value
void _cs348v_vabs_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_mask &mask);
void _cs348v_vabs_int(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_mask &mask);

// Return a mask of (veca > vecb) if vector lane active
//  otherwise keep the old value
void _cs348v_vgt_float(__cs348v_mask &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
void _cs348v_vgt_int(__cs348v_mask &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

// Return a mask of (veca < vecb) if vector lane active
//  otherwise keep the old value
void _cs348v_vlt_float(__cs348v_mask &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
void _cs348v_vlt_int(__cs348v_mask &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

// Return a mask of (veca == vecb) if vector lane active
//  otherwise keep the old value
void _cs348v_veq_float(__cs348v_mask &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
void _cs348v_veq_int(__cs348v_mask &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

// Adds up adjacent pairs of elements, so
//  [0 1 2 3] -> [0+1 0+1 2+3 2+3]
void _cs348v_hadd_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &vec);

// Performs an even-odd interleaving where all even-indexed elements move to front half
//  of the array and odd-indexed to the back half, so
//  [0 1 2 3 4 5 6 7] -> [0 2 4 6 1 3 5 7]
void _cs348v_interleave_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &vec);

// Add a customized log to help debugging
void addUserLog(const char * logStr);

#endif
