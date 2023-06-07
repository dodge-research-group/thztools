import h5py  # type: ignore
import numpy as np
import pathlib
import pytest

# import matplotlib.pyplot as plt
from thztools import (
    costfunlsq,
    epswater,
    fftfreq,
    noiseamp,
    noisevar,
    shiftmtx,
    tdnll,
    tdnoisefit, # TODO
    tdtf,
    thzgen,
)


def tdnll_alt(*args):
    x = args[0]
    param = {'logv': args[1],
             'mu': args[2],
             'a': args[3],
             'eta': args[4],
             'ts': args[5],
             'D': args[6]}
    fix = {'logv': False,
           'mu': False,
           'a': False,
           'eta': False}
    return tdnll(x, param, fix)


# Establish dictionary mapping from function names to functions
FUNC_DICT = {"fftfreq": fftfreq, "epswater": epswater, "thzgen": thzgen,
             'noisevar': noisevar, 'noiseamp': noiseamp, 'shiftmtx': shiftmtx,
             'tdtf': tdtf, 'costfunlsq': costfunlsq, 'tdnll': tdnll_alt}

# Set MAT-file path
cur_path = pathlib.Path(__file__).parents[1].resolve()
f_path = cur_path / "matlab" / "thztools_test_data.mat"


def tfun_test(_theta, _w):
    return _theta[0] * np.exp(-1j * _w * _theta[1])


# Read test array from MAT-file
def get_matlab_tests():
    with h5py.File(f_path, "r") as f_obj:
        # The MAT-file stores a structure named "Set" with a field for each
        # function in the test set. Get the field names (which are also the
        # function names) and loop over them.
        func_names = list(f_obj['Set'].keys())
        test_list = []
        for func_name in func_names:
            # The MATLAB inputs and outputs for each test configuration are
            # stored in the HDF5 dataset arrays "args" and "out", respectively.
            # The "[()]" index converts the HDF5 dataset arrays to NumPy
            # arrays for easier manipulation, such as flattening.
            arg_refs = f_obj['Set'][func_name]['args'][()].flatten()
            out_refs = f_obj['Set'][func_name]['out'][()].flatten()
            # Get the elements of the "args" and "out" arrays and eliminate
            # extraneous array dimensions.
            for arg_ref, out_ref in zip(arg_refs, out_refs):
                args_val_list = []
                out_val_list = []
                arg_val_refs = f_obj[arg_ref][()].flatten()
                for arg_val_ref in arg_val_refs:
                    # MATLAB apparently writes 2D arrays to HDF5 files as
                    # transposed C-order arrays, so we need to transpose them
                    # back after reading them in.
                    arg_val = np.squeeze(f_obj[arg_val_ref][()]).T
                    # Convert scalar arrays to scalars
                    if arg_val.shape == ():
                        arg_val = arg_val[()]
                    args_val_list.append(arg_val)
                out_val_refs = f_obj[out_ref][()].flatten()
                for out_val_ref in out_val_refs:
                    # MATLAB apparently writes 2D arrays to HDF5 files as
                    # transposed C-order arrays, so we need to transpose them
                    # back after reading them in.
                    out_val = np.squeeze(f_obj[out_val_ref][()]).T
                    # Convert scalar arrays to scalars
                    if out_val.shape == ():
                        out_val = out_val[()]
                    # The MAT-file stores complex numbers as tuples with a
                    # composite dtype. Convert these to NumPy complex dtypes.
                    if (out_val.dtype.names is not None
                            and 'real' in out_val.dtype.names
                            and 'imag' in out_val.dtype.names):
                        out_val = out_val['real'] + 1j * out_val['imag']
                    out_val_list.append(out_val)
                test_list.append([func_name, args_val_list, out_val_list])
    return test_list


@pytest.fixture(params=get_matlab_tests())
def get_test(request):
    return request.param


def test_matlab_result(get_test):
    func_name = get_test[0]
    func = FUNC_DICT[func_name]
    args = get_test[1]
    matlab_out = get_test[2]
    if func_name in ['tdtf', 'costfunlsq']:
        python_out = func(tfun_test, *args)
    else:
        python_out = func(*args)
    # Ignore second output from Python version of thzgen
    if func_name in ["thzgen", "tdnll"]:
        python_out = python_out[0]
    # Set absolute tolerance equal to 2 * epsilon for the array dtype
    np.testing.assert_allclose(matlab_out[0], python_out,
                               atol=2 * np.finfo(python_out.dtype).eps)

# def test_tdnoisefit():
#     cur_path = pathlib.Path(__file__).parent.resolve()
#     fname = cur_path / "test_files" / "tdnoisefit_test_data.mat"
#     with h5py.File(fname, "r") as file:
#         file_set = file["Set"]
#         # x = file['Set']['tdnoisefit']['x'][0]
#         param = file["Set"]["tdnoisefit"]["paramForPy"]
#         fix = file["Set"]["tdnoisefit"]["fixForPy"]
#         ignore = file["Set"]["tdnoisefit"]["ignoreForPy"]
#         p_matlab = file["Set"]["tdnoisefit"]["P"]
#         # diagnostic = file['Set']['tdnoisefit']['Diagnostic']
#         # dfx = np.array(x)
#         # dfParam = np.array(param)
#
#         # for i in range(0, dfx.shape[0]):
#         # for j in range(0, dfParam.shape[0]):
#         for i in range(0, 1):
#             for j in range(0, 1):
#                 x = np.array(file[file_set["tdnoisefit"]["x"][j, i]]).T
#                 param = {
#                     "v0": np.array(file[param[j, 0]]["v0"])[0],
#                     "mu0": np.array(file[param[j, 0]]["mu0"])[0],
#                     "a0": np.array(file[param[j, 0]]["A0"])[0],
#                     "eta0": np.array(file[param[j, 0]]["eta0"])[0],
#                     "ts": np.array(file[param[j, 0]]["ts"])[0],
#                 }
#                 fix = {
#                     "logv": bool(np.array(file[fix[j, 0]]["logv"])),
#                     "mu": bool(np.array(file[fix[j, 0]]["mu"])),
#                     "a": bool(np.array(file[fix[j, 0]]["A"])),
#                     "eta": bool(np.array(file[fix[j, 0]]["eta"])),
#                 }
#                 ignore = {
#                     "a": bool(np.array(file[ignore[j, 0]]["A"])),
#                     "eta": bool(np.array(file[ignore[j, 0]]["eta"])),
#                 }
#                 p = {
#                     "var": np.array(file[p_matlab[j, 0]]["var"])[0],
#                     "mu": np.array(file[p_matlab[j, 0]]["mu"])[0],
#                     "a": np.array(file[p_matlab[j, 0]]["A"])[0],
#                     "eta": np.array(file[p_matlab[j, 0]]["eta"])[0],
#                     "ts": np.array(file[p_matlab[j, 0]]["ts"])[0],
#                 }
#                 fun = np.array(file[file_set["tdnoisefit"]["fval"][j, i]])[0]
#                 # diagnostic = np.array(
#                 # file[func_set['tdnoisefit']['Diagnostic'][j, i]])[0]
#                 [p_py, fun_py, _] = tdnoisefit(x, param, fix, ignore)
#                 print("Matlab Costfun: " + str(fun))
#                 print("Python Costfun: " + str(fun_py))
#                 print("-------------------")
#                 print("Matlab var: " + str(p["var"]))
#                 print("Python var: " + str(p_py["var"]))
#                 np.testing.assert_allclose(funPy, fun, atol=1e-02)
#                 np.testing.assert_allclose(pPy['var'], p['var'], atol=1e-02)
#                 np.testing.assert_allclose(pPy['mu'], p['mu'], atol=1e-02)
#                 np.testing.assert_allclose(pPy['a'], p['a'], atol=1e-02)
#                 np.testing.assert_allclose(pPy['eta'], p['eta'], atol=1e-02)
#                 np.testing.assert_allclose(pPy['ts'], p['ts'], atol=1e-02)
#                 np.testing.assert_allclose(diagnosticPy, diagnostic)
