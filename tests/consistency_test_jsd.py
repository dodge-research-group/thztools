import h5py  # type: ignore
import numpy as np
import pathlib
import pytest

# import matplotlib.pyplot as plt
from thztools import (
    costfunlsq,
    epswater,
    fftfreq,
    noisevar,
    tdnll,
    tdnoisefit,
    tdtf,
    thzgen,
)


FUNC_DICT = {"fftfreq": fftfreq}
cur_path = pathlib.Path(__file__).parents[1].resolve()
fname = cur_path / "matlab" / "thztools_test_data.mat"
print(fname)


def get_param_list(set_name):
    # Initiate parameter list
    param_list = []
    with h5py.File(fname, 'r') as f:
        # Get HDF5 group for original MATLAB structure array
        f_set = f["Set"][set_name]
        # Get structure field names
        k = list(f_set.keys())
        # Loop through field names
        for item in k:
            # Get pointers to variables associated with a given field
            refs = np.array(f_set[item]).flatten()
            var_list = []
            # Include variables for each pointer in var_list
            for ref in refs:
                var = np.squeeze(f[ref][()])
                if var.shape == ():
                    var = var[()]
                var_list.append(var)
            # Append variable list for the current field
            param_list.append(var_list)
        # Transpose param_list for sequential calls from test
        param_list = list(zip(*param_list))
    return param_list


def get_data_dict():
    data_dict = dict()
    with h5py.File(fname, 'r') as f:
        # Get function names
        func_list = list(f["Set"].keys())
        # Loop over functions
        for func_name in func_list:
            # Initiate parameter list
            param_list = []
            # Get HDF5 group for original MATLAB structure array
            func_group = f["Set"][func_name]
            # Get structure field names
            field_list = list(func_group.keys())
            # Loop through field names
            for field in field_list:
                # Get pointers to variables associated with a given field
                refs = np.array(func_group[field]).flatten()
                var_list = []
                # Include variable at each pointer in var_list
                for ref in refs:
                    var = np.squeeze(f[ref][()])
                    if var.shape == ():
                        var = var[()]
                    var_list.append(var)
                # Append variable list for the current field
                param_list.append(var_list)
            # Transpose param_list for sequential calls from test
            param_list = list(zip(*param_list))
            data_dict[func_name] = param_list
    return data_dict


def get_matlab_tests():
    results = []
    with h5py.File(fname, 'r') as f:
        # Get function names
        func_list = list(f["Set"].keys())
        # Loop over functions
        for func_name in func_list:
            # Initiate parameter list
            param_list = []
            # Get HDF5 group for original MATLAB structure array
            func_group = f["Set"][func_name]
            # Get structure field names
            field_list = list(func_group.keys())
            # Loop through field names
            for field in field_list:
                # Get pointers to variables associated with a given field
                refs = np.array(func_group[field]).flatten()
                var_list = []
                # Include variable at each pointer in var_list
                for ref in refs:
                    var = np.squeeze(f[ref][()])
                    if var.shape == ():
                        var = var[()]
                    var_list.append(var)
                # Append variable list for the current field
                param_list.append(var_list)
            # Transpose param_list for sequential calls from test
            param_list = list(zip([func_name] * len(param_list[0]),
                                  *param_list))
            results.append(param_list)
    return results


# @pytest.fixture(params=get_data_dict())
# def test_func(request):
#     return request.param


# @pytest.fixture(params=get_param_list("fftfreq"))
# def get_data(request):
#     return request.param
#
#
# def test_fftfreq(get_params):
#     n = get_params[0]
#     t = get_params[1]
#     f = get_params[2]
#     fpy = fftfreq(int(n), t)
#     np.testing.assert_allclose(f, fpy)
#
#
# def test_fftfreq():
#     cur_path = pathlib.Path(__file__).parent.resolve()
#     fname = cur_path / "test_files" / "fftfreq_test_data.mat"
#
#     with h5py.File(fname, "r") as file:
#         # Get HDF5 group for original MATLAB structure array
#         f_set = file["Set"]["fftfreq"]
#         # Each field has the same size; use one to get dimensions
#         ni, nj = f_set["N"].shape
#
#         # Test all values in the array
#         for i in range(0, ni):
#             for j in range(0, nj):
#                 # The elements of f_set are object references that point to
#                 # the values in the MATLAB structure array, so to access them
#                 # we use the syntax file[obj_ref], and to convert the HDF5
#                 # dataset object to a ndarray we use file[obj_ref][()].
#                 # MATLAB uses 2D arrays for scalars and vectors, so apply
#                 # np.squeeze to reduce dimensions.
#                 n = np.squeeze(file[f_set["N"][i, j]][()])
#                 t = np.squeeze(file[f_set["T"][i, j]][()])
#                 y = np.squeeze(file[f_set["f"][i, j]][()])
#                 # Convert n to integer type for fftfreq
#                 n = int(n)
#                 # Compute fftfreq and compare with MATLAB output
#                 fpy = fftfreq(n, t)
#                 np.testing.assert_allclose(y, fpy)
#
#
# def test_noisevar():
#     cur_path = pathlib.Path(__file__).parent.resolve()
#     fname = cur_path / "test_files" / "noisevar_test_data.mat"
#     with h5py.File(fname, "r") as file:
#         file_set = file["Set"]
#         sigma = file["Set"]["noisevar"]["sigma"][0]
#         mu = file["Set"]["noisevar"]["mu"][0]
#         t = file["Set"]["noisevar"]["T"][0]
#         # vmu = file['Set']['noisevar']['Vmu'][0]
#         nsigma = np.array(sigma)
#         nmu = np.array(mu)
#         nt = np.array(t)
#
#         for i in range(0, nsigma.shape[0]):
#             for j in range(0, nmu.shape[0]):
#                 for k in range(0, nt.shape[0]):
#                     sigma = np.array(
#                         file[file_set["noisevar"]["sigma"][i, j, k]]
#                     )[0]
#                     mu = np.array(file[file_set["noisevar"]["mu"][i, j, k]])[0]
#                     t = np.array(file[file_set["noisevar"]["T"][i, j, k]])[
#                         0, 0
#                     ]
#                     vmu = np.array(file[file_set["noisevar"]["Vmu"][i, j, k]])[
#                         0
#                     ]
#                     vmupy = noisevar(sigma, mu, t)
#                     np.testing.assert_allclose(vmu, vmupy)
#
#
# def test_epswater():
#     cur_path = pathlib.Path(__file__).parent.resolve()
#     fname = cur_path / "test_files" / "epswater_test_data.mat"
#     with h5py.File(fname, "r") as file:
#         file_set = file["Set"]
#         xf = file["Set"]["epswater"]["f"][0]
#         xt = file["Set"]["epswater"]["T"][0]
#         dff = np.array(xf)
#         dft = np.array(xt)
#
#         for i in range(0, dff.shape[0]):
#             for j in range(0, dft.shape[0]):
#                 f = np.array(file[file_set["epswater"]["f"][i, j]])[0, 0]
#                 t = np.array(file[file_set["epswater"]["T"][i, j]])[0, 0]
#                 epsr = np.array(file[file_set["epswater"]["epsR"][i, j]])
#                 epsi = np.array(file[file_set["epswater"]["epsI"][i, j]])
#                 fpy = epswater(f, t)
#                 np.testing.assert_allclose(epsr, np.real(fpy))
#                 np.testing.assert_allclose(epsi, np.imag(fpy))
#
#
# def test_thzgen():
#     cur_path = pathlib.Path(__file__).parent.resolve()
#     fname = cur_path / "test_files" / "thzgen_test_data.mat"
#
#     with h5py.File(fname, "r") as file:
#         file_set = file["Set"]
#         xn = file["Set"]["thzgen"]["N"][0]
#         xt = file["Set"]["thzgen"]["T"][0]
#         xt0 = file["Set"]["thzgen"]["t0"][0]
#         dfn = np.array(xn)
#         dft = np.array(xt)
#         dft0 = np.array(xt0)
#
#         for i in range(0, dfn.shape[0]):
#             for j in range(0, dft.shape[0]):
#                 for k in range(0, dft0.shape[0]):
#                     n = np.array(file[file_set["thzgen"]["N"][i, j, k]])[0, 0]
#                     t = np.array(file[file_set["thzgen"]["T"][i, j, k]])[0, 0]
#                     t0 = np.array(file[file_set["thzgen"]["t0"][i, j, k]])[
#                         0, 0
#                     ]
#                     y = np.array(file[file_set["thzgen"]["y"][i, j, k]])[0]
#                     fpy = thzgen(n.astype(int), t, t0)[0]
#                     np.testing.assert_allclose(y, fpy)
#
#
# def test_costfunlsq():
#     cur_path = pathlib.Path(__file__).parent.resolve()
#     fname = cur_path / "test_files" / "costfunlsq_test_data.mat"
#     with h5py.File(fname, "r") as file:
#         file_set = file["Set"]
#         xtheta = file["Set"]["costfunlsq"]["theta"][0]
#         x_xx = file["Set"]["costfunlsq"]["xx"][0]
#         x_yy = file["Set"]["costfunlsq"]["yy"][0]
#         xsigmax = file["Set"]["costfunlsq"]["sigmax"][0]
#         xsigmay = file["Set"]["costfunlsq"]["sigmay"][0]
#         xwfft = file["Set"]["costfunlsq"]["wfft"][0]
#         dftheta = np.array(xtheta)
#         dfxx = np.array(x_xx)
#         dfyy = np.array(x_yy)
#         dfsigmax = np.array(xsigmax)
#         dfsigmay = np.array(xsigmay)
#         dfwfft = np.array(xwfft)
#
#         def fun(_theta, _wfft):
#             return _theta[0] * np.exp(1j * _theta[1] * _wfft)
#
#         for i in range(0, dftheta.shape[0]):
#             for j in range(0, dfxx.shape[0]):
#                 for k in range(0, dfyy.shape[0]):
#                     for m in range(0, dfsigmax.shape[0]):
#                         for n in range(0, dfsigmay.shape[0]):
#                             for p in range(0, dfwfft.shape[0]):
#                                 theta = np.array(
#                                     file[
#                                         file_set["costfunlsq"]["theta"][
#                                             i, j, k, m, n, p
#                                         ]
#                                     ]
#                                 )[0]
#                                 xx = np.array(
#                                     file[
#                                         file_set["costfunlsq"]["xx"][
#                                             i, j, k, m, n, p
#                                         ]
#                                     ]
#                                 )[0]
#                                 yy = np.array(
#                                     file[
#                                         file_set["costfunlsq"]["yy"][
#                                             i, j, k, m, n, p
#                                         ]
#                                     ]
#                                 )[0]
#                                 sigmax = np.array(
#                                     file[
#                                         file_set["costfunlsq"]["sigmax"][
#                                             i, j, k, m, n, p
#                                         ]
#                                     ]
#                                 )[0]
#                                 sigmay = np.array(
#                                     file[
#                                         file_set["costfunlsq"]["sigmay"][
#                                             i, j, k, m, n, p
#                                         ]
#                                     ]
#                                 )[0]
#                                 # wfft = np.array(
#                                 # file[Set['costfunlsq'][
#                                 # 'wfft'][i, j, k, l, m, n]])[0]
#                                 res = np.array(
#                                     file[
#                                         file_set["costfunlsq"]["res"][
#                                             i, j, k, m, n, p
#                                         ]
#                                     ]
#                                 )[0]
#                                 fpy = costfunlsq(
#                                     fun, theta, xx, yy, sigmax, sigmay, i + 1
#                                 )
#                                 np.testing.assert_allclose(res, fpy)
#
#
# def test_tdtf():
#     def fun(_theta, _w):
#         return _theta[0] * np.exp(-1j * _theta[1] * _w)
#
#     cur_path = pathlib.Path(__file__).parent.resolve()
#     fname = cur_path / "test_files" / "tdtf_test_data.mat"
#
#     with h5py.File(fname, "r") as file:
#         file_set = file["Set"]
#         theta = file["Set"]["tdtf"]["theta"][0]
#         n = file["Set"]["tdtf"]["N"][0]
#         ts = file["Set"]["tdtf"]["ts"][0]
#
#         ntheta = np.array(theta)
#         nn = np.array(n)
#         nts = np.array(ts)
#
#         for i in range(0, ntheta.shape[0]):
#             for j in range(0, nn.shape[0]):
#                 for k in range(0, nts.shape[0]):
#                     theta = np.array(file[file_set["tdtf"]["theta"][i, j, k]])[
#                         0
#                     ]
#                     n = np.array(file[file_set["tdtf"]["N"][i, j, k]])[0, 0]
#                     ts = np.array(file[file_set["tdtf"]["ts"][i, j, k]])[0, 0]
#                     h = np.array(file[file_set["tdtf"]["h"][i, j, k]])
#                     hpy = tdtf(fun, theta, n, ts)
#                     h = np.transpose(h)
#                     np.testing.assert_allclose(hpy, h, atol=1e-10)
#
#
# def test_tdnll():
#     cur_path = pathlib.Path(__file__).parent.resolve()
#     fname = cur_path / "test_files" / "tdnll_test_data.mat"
#     with h5py.File(fname, "r") as file:
#         file_set = file["Set"]
#         x = file["Set"]["tdnll"]["x"][0]
#         p_struct = file["Set"]["tdnll"]["Param"]
#         varargin_struct = file["Set"]["tdnll"]["varargin"]
#         dfx = np.array(x)
#         df_param = np.array(p_struct)
#         dfvarargin = np.array(varargin_struct)
#
#         for i in range(0, dfx.shape[1]):
#             for j in range(0, df_param.shape[0]):
#                 for k in range(0, dfvarargin.shape[0]):
#                     x = np.array(file[file_set["tdnll"]["x"][k, j, i]]).T
#                     param = {
#                         "logv": np.array(file[p_struct[0, j, 0]]["logv"])[0],
#                         "mu": np.array(file[p_struct[0, j, 0]]["mu"])[0],
#                         "a": np.array(file[p_struct[0, j, 0]]["A"])[0],
#                         "eta": np.array(file[p_struct[0, j, 0]]["eta"])[0],
#                         "ts": np.array(file[p_struct[0, j, 0]]["ts"])[0],
#                         "D": np.array(file[p_struct[0, j, 0]]["D"]),
#                     }
#                     varargin = {
#                         "logv": bool(
#                             np.array(file[varargin_struct[k, 0, 0]]["logv"])
#                         ),
#                         "mu": bool(
#                             np.array(file[varargin_struct[k, 0, 0]]["mu"])
#                         ),
#                         "a": bool(
#                             np.array(file[varargin_struct[k, 0, 0]]["A"])
#                         ),
#                         "eta": bool(
#                             np.array(file[varargin_struct[k, 0, 0]]["eta"])
#                         ),
#                     }
#                     nll = np.array(file[file_set["tdnll"]["nll"][k, j, i]])[
#                         0, 0
#                     ]
#                     # gradnll = np.array(
#                     #     file[func_set['tdnll']['gradnll'][k, j, i]])[0]
#                     [nll_py, _] = tdnll(x, param, varargin)
#                     np.testing.assert_allclose(nll_py, nll)
#                     # np.testing.assert_allclose(gradnllPy, gradnll, rtol=1e-3)
#
#
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
