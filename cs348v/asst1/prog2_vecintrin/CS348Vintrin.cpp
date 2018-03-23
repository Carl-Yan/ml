#include "CS348Vintrin.h"
#include "logger.h"

//******************
//* Implementation *
//******************

__cs348v_mask _cs348v_init_ones(int first) {
  __cs348v_mask mask;
  for (int i=0; i<VECTOR_WIDTH; i++) {
    mask.value[i] = (i<first) ? true : false;
  }
  return mask;
}

__cs348v_mask _cs348v_mask_not(__cs348v_mask &maska) {
  __cs348v_mask resultMask;
  for (int i=0; i<VECTOR_WIDTH; i++) {
    resultMask.value[i] = !maska.value[i];
  }
  CS348VLogger.addLog("masknot", _cs348v_init_ones(), VECTOR_WIDTH);
  return resultMask;
}

__cs348v_mask _cs348v_mask_or(__cs348v_mask &maska, __cs348v_mask &maskb) {
  __cs348v_mask resultMask;
  for (int i=0; i<VECTOR_WIDTH; i++) {
    resultMask.value[i] = maska.value[i] | maskb.value[i];
  }
  CS348VLogger.addLog("maskor", _cs348v_init_ones(), VECTOR_WIDTH);
  return resultMask;
}

__cs348v_mask _cs348v_mask_and(__cs348v_mask &maska, __cs348v_mask &maskb) {
  __cs348v_mask resultMask;
  for (int i=0; i<VECTOR_WIDTH; i++) {
    resultMask.value[i] = maska.value[i] && maskb.value[i];
  }
  CS348VLogger.addLog("maskand", _cs348v_init_ones(), VECTOR_WIDTH);
  return resultMask;
}

int _cs348v_cntbits(__cs348v_mask &maska) {
  int count = 0;
  for (int i=0; i<VECTOR_WIDTH; i++) {
    if (maska.value[i]) count++;
  }
  CS348VLogger.addLog("cntbits", _cs348v_init_ones(), VECTOR_WIDTH);
  return count;
}

template <typename T>
void _cs348v_vset(__cs348v_vec<T> &vecResult, T value, __cs348v_mask &mask) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    vecResult.value[i] = mask.value[i] ? value : vecResult.value[i];
  }
  CS348VLogger.addLog("vset", mask, VECTOR_WIDTH);
}

template void _cs348v_vset<float>(__cs348v_vec_float &vecResult, float value, __cs348v_mask &mask);
template void _cs348v_vset<int>(__cs348v_vec_int &vecResult, int value, __cs348v_mask &mask);

void _cs348v_vset_float(__cs348v_vec_float &vecResult, float value, __cs348v_mask &mask) { _cs348v_vset<float>(vecResult, value, mask); }
void _cs348v_vset_int(__cs348v_vec_int &vecResult, int value, __cs348v_mask &mask) { _cs348v_vset<int>(vecResult, value, mask); }

__cs348v_vec_float _cs348v_vset_float(float value) {
  __cs348v_vec_float vecResult;
  __cs348v_mask mask = _cs348v_init_ones();
  _cs348v_vset_float(vecResult, value, mask);
  return vecResult;
}
__cs348v_vec_int _cs348v_vset_int(int value) {
  __cs348v_vec_int vecResult;
  __cs348v_mask mask = _cs348v_init_ones();
  _cs348v_vset_int(vecResult, value, mask);
  return vecResult;
}

template <typename T>
void _cs348v_vmove(__cs348v_vec<T> &dest, __cs348v_vec<T> &src, __cs348v_mask &mask) {
    for (int i = 0; i < VECTOR_WIDTH; i++) {
        dest.value[i] = mask.value[i] ? src.value[i] : dest.value[i];
    }
    CS348VLogger.addLog("vmove", mask, VECTOR_WIDTH);
}

template void _cs348v_vmove<float>(__cs348v_vec_float &dest, __cs348v_vec_float &src, __cs348v_mask &mask);
template void _cs348v_vmove<int>(__cs348v_vec_int &dest, __cs348v_vec_int &src, __cs348v_mask &mask);

void _cs348v_vmove_float(__cs348v_vec_float &dest, __cs348v_vec_float &src, __cs348v_mask &mask) { _cs348v_vmove<float>(dest, src, mask); }
void _cs348v_vmove_int(__cs348v_vec_int &dest, __cs348v_vec_int &src, __cs348v_mask &mask) { _cs348v_vmove<int>(dest, src, mask); }

template <typename T>
void _cs348v_vload(__cs348v_vec<T> &dest, T* src, __cs348v_mask &mask) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    dest.value[i] = mask.value[i] ? src[i] : dest.value[i];
  }
  CS348VLogger.addLog("vload", mask, VECTOR_WIDTH);
}

template void _cs348v_vload<float>(__cs348v_vec_float &dest, float* src, __cs348v_mask &mask);
template void _cs348v_vload<int>(__cs348v_vec_int &dest, int* src, __cs348v_mask &mask);

void _cs348v_vload_float(__cs348v_vec_float &dest, float* src, __cs348v_mask &mask) { _cs348v_vload<float>(dest, src, mask); }
void _cs348v_vload_int(__cs348v_vec_int &dest, int* src, __cs348v_mask &mask) { _cs348v_vload<int>(dest, src, mask); }

template <typename T>
void _cs348v_vstore(T* dest, __cs348v_vec<T> &src, __cs348v_mask &mask) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    dest[i] = mask.value[i] ? src.value[i] : dest[i];
  }
  CS348VLogger.addLog("vstore", mask, VECTOR_WIDTH);
}

template void _cs348v_vstore<float>(float* dest, __cs348v_vec_float &src, __cs348v_mask &mask);
template void _cs348v_vstore<int>(int* dest, __cs348v_vec_int &src, __cs348v_mask &mask);

void _cs348v_vstore_float(float* dest, __cs348v_vec_float &src, __cs348v_mask &mask) { _cs348v_vstore<float>(dest, src, mask); }
void _cs348v_vstore_int(int* dest, __cs348v_vec_int &src, __cs348v_mask &mask) { _cs348v_vstore<int>(dest, src, mask); }

template <typename T>
void _cs348v_vadd(__cs348v_vec<T> &vecResult, __cs348v_vec<T> &veca, __cs348v_vec<T> &vecb, __cs348v_mask &mask) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    vecResult.value[i] = mask.value[i] ? (veca.value[i] + vecb.value[i]) : vecResult.value[i];
  }
  CS348VLogger.addLog("vadd", mask, VECTOR_WIDTH);
}

template void _cs348v_vadd<float>(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
template void _cs348v_vadd<int>(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

void _cs348v_vadd_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask) { _cs348v_vadd<float>(vecResult, veca, vecb, mask); }
void _cs348v_vadd_int(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask) { _cs348v_vadd<int>(vecResult, veca, vecb, mask); }

template <typename T>
void _cs348v_vsub(__cs348v_vec<T> &vecResult, __cs348v_vec<T> &veca, __cs348v_vec<T> &vecb, __cs348v_mask &mask) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    vecResult.value[i] = mask.value[i] ? (veca.value[i] - vecb.value[i]) : vecResult.value[i];
  }
  CS348VLogger.addLog("vsub", mask, VECTOR_WIDTH);
}

template void _cs348v_vsub<float>(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
template void _cs348v_vsub<int>(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

void _cs348v_vsub_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask) { _cs348v_vsub<float>(vecResult, veca, vecb, mask); }
void _cs348v_vsub_int(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask) { _cs348v_vsub<int>(vecResult, veca, vecb, mask); }

template <typename T>
void _cs348v_vmult(__cs348v_vec<T> &vecResult, __cs348v_vec<T> &veca, __cs348v_vec<T> &vecb, __cs348v_mask &mask) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    vecResult.value[i] = mask.value[i] ? (veca.value[i] * vecb.value[i]) : vecResult.value[i];
  }
  CS348VLogger.addLog("vmult", mask, VECTOR_WIDTH);
}

template void _cs348v_vmult<float>(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
template void _cs348v_vmult<int>(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

void _cs348v_vmult_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask) { _cs348v_vmult<float>(vecResult, veca, vecb, mask); }
void _cs348v_vmult_int(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask) { _cs348v_vmult<int>(vecResult, veca, vecb, mask); }

template <typename T>
void _cs348v_vdiv(__cs348v_vec<T> &vecResult, __cs348v_vec<T> &veca, __cs348v_vec<T> &vecb, __cs348v_mask &mask) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    vecResult.value[i] = mask.value[i] ? (veca.value[i] / vecb.value[i]) : vecResult.value[i];
  }
  CS348VLogger.addLog("vdiv", mask, VECTOR_WIDTH);
}

template void _cs348v_vdiv<float>(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
template void _cs348v_vdiv<int>(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

void _cs348v_vdiv_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask) { _cs348v_vdiv<float>(vecResult, veca, vecb, mask); }
void _cs348v_vdiv_int(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask) { _cs348v_vdiv<int>(vecResult, veca, vecb, mask); }

template <typename T>
void _cs348v_vabs(__cs348v_vec<T> &vecResult, __cs348v_vec<T> &veca, __cs348v_mask &mask) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    vecResult.value[i] = mask.value[i] ? (abs(veca.value[i])) : vecResult.value[i];
  }
  CS348VLogger.addLog("vabs", mask, VECTOR_WIDTH);
}

template void _cs348v_vabs<float>(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_mask &mask);
template void _cs348v_vabs<int>(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_mask &mask);

void _cs348v_vabs_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &veca, __cs348v_mask &mask) { _cs348v_vabs<float>(vecResult, veca, mask); }
void _cs348v_vabs_int(__cs348v_vec_int &vecResult, __cs348v_vec_int &veca, __cs348v_mask &mask) { _cs348v_vabs<int>(vecResult, veca, mask); }

template <typename T>
void _cs348v_vgt(__cs348v_mask &maskResult, __cs348v_vec<T> &veca, __cs348v_vec<T> &vecb, __cs348v_mask &mask) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    maskResult.value[i] = mask.value[i] ? (veca.value[i] > vecb.value[i]) : maskResult.value[i];
  }
  CS348VLogger.addLog("vgt", mask, VECTOR_WIDTH);
}

template void _cs348v_vgt<float>(__cs348v_mask &maskResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
template void _cs348v_vgt<int>(__cs348v_mask &maskResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

void _cs348v_vgt_float(__cs348v_mask &maskResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask) { _cs348v_vgt<float>(maskResult, veca, vecb, mask); }
void _cs348v_vgt_int(__cs348v_mask &maskResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask) { _cs348v_vgt<int>(maskResult, veca, vecb, mask); }

template <typename T>
void _cs348v_vlt(__cs348v_mask &maskResult, __cs348v_vec<T> &veca, __cs348v_vec<T> &vecb, __cs348v_mask &mask) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    maskResult.value[i] = mask.value[i] ? (veca.value[i] < vecb.value[i]) : maskResult.value[i];
  }
  CS348VLogger.addLog("vlt", mask, VECTOR_WIDTH);
}

template void _cs348v_vlt<float>(__cs348v_mask &maskResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
template void _cs348v_vlt<int>(__cs348v_mask &maskResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

void _cs348v_vlt_float(__cs348v_mask &maskResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask) { _cs348v_vlt<float>(maskResult, veca, vecb, mask); }
void _cs348v_vlt_int(__cs348v_mask &maskResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask) { _cs348v_vlt<int>(maskResult, veca, vecb, mask); }

template <typename T>
void _cs348v_veq(__cs348v_mask &maskResult, __cs348v_vec<T> &veca, __cs348v_vec<T> &vecb, __cs348v_mask &mask) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    maskResult.value[i] = mask.value[i] ? (veca.value[i] == vecb.value[i]) : maskResult.value[i];
  }
  CS348VLogger.addLog("veq", mask, VECTOR_WIDTH);
}

template void _cs348v_veq<float>(__cs348v_mask &maskResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask);
template void _cs348v_veq<int>(__cs348v_mask &maskResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask);

void _cs348v_veq_float(__cs348v_mask &maskResult, __cs348v_vec_float &veca, __cs348v_vec_float &vecb, __cs348v_mask &mask) { _cs348v_veq<float>(maskResult, veca, vecb, mask); }
void _cs348v_veq_int(__cs348v_mask &maskResult, __cs348v_vec_int &veca, __cs348v_vec_int &vecb, __cs348v_mask &mask) { _cs348v_veq<int>(maskResult, veca, vecb, mask); }

template <typename T>
void _cs348v_hadd(__cs348v_vec<T> &vecResult, __cs348v_vec<T> &vec) {
  for (int i=0; i<VECTOR_WIDTH/2; i++) {
    T result = vec.value[2*i] + vec.value[2*i+1];
    vecResult.value[2 * i] = result;
    vecResult.value[2 * i + 1] = result;
  }
}

template void _cs348v_hadd<float>(__cs348v_vec_float &vecResult, __cs348v_vec_float &vec);

void _cs348v_hadd_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &vec) { _cs348v_hadd<float>(vecResult, vec); }

template <typename T>
void _cs348v_interleave(__cs348v_vec<T> &vecResult, __cs348v_vec<T> &vec) {
  for (int i=0; i<VECTOR_WIDTH; i++) {
    int index = i < VECTOR_WIDTH/2 ? (2 * i) : (2 * (i - VECTOR_WIDTH/2) + 1);
    vecResult.value[i] = vec.value[index];
  }
}

template void _cs348v_interleave<float>(__cs348v_vec_float &vecResult, __cs348v_vec_float &vec);

void _cs348v_interleave_float(__cs348v_vec_float &vecResult, __cs348v_vec_float &vec) { _cs348v_interleave<float>(vecResult, vec); }

void addUserLog(const char * logStr) {
  CS348VLogger.addLog(logStr, _cs348v_init_ones(), 0);
}

