import pathlib

import h5py  # type: ignore
import numpy as np
import pytest

from thztools import (
    costfunlsq,
    fftfreq,
    noiseamp,
    noisevar,
    tdnll,
    tdnoisefit,
    tdtf,
    thzgen,
)
# np.seterr(all='raise')


def tdnll_alt(*args):
    kwargs = {
        "fix_logv": False,
        "fix_mu": False,
        "fix_a": False,
        "fix_eta": False,
    }
    return tdnll(*args, **kwargs)


def tdnoisefit_alt(*args):
    kwargs = {
        "fix_v": False,
        "fix_mu": False,
        "fix_a": False,
        "fix_eta": False,
        "ignore_a": True,
        "ignore_eta": False,
    }
    out, _, _ = tdnoisefit(*args, **kwargs)
    out_alt = [out["var"], out["mu"], out["a"], out["eta"]]
    return out_alt


# Establish dictionary mapping from function names to functions
FUNC_DICT = {
    "fftfreq": fftfreq,
    "thzgen": thzgen,
    "noisevar": noisevar,
    "noiseamp": noiseamp,
    "tdtf": tdtf,
    "costfunlsq": costfunlsq,
    "tdnll": tdnll_alt,
    "tdnoisefit": tdnoisefit_alt,
}

# Set MAT-file path
cur_path = pathlib.Path(__file__).parents[0].resolve()
f_path = cur_path / "matlab" / "thztools_test_data.mat"


def tfun_test(_theta, _w):
    return _theta[0] * np.exp(-1j * _w * _theta[1])


# Read test array from MAT-file
def get_matlab_tests():
    with h5py.File(f_path, "r") as f_obj:
        # The MAT-file stores a structure named "Set" with a field for each
        # function in the test set. Get the field names (which are also the
        # function names) and loop over them.
        func_names = list(f_obj["Set"].keys())
        test_list = []
        for func_name in func_names:
            # The MATLAB inputs and outputs for each test configuration are
            # stored in the HDF5 dataset arrays "args" and "out", respectively.
            # The "[()]" index converts the HDF5 dataset arrays to NumPy
            # arrays for easier manipulation, such as flattening.
            arg_refs = f_obj["Set"][func_name]["args"][()].flatten()
            out_refs = f_obj["Set"][func_name]["out"][()].flatten()
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
                    if (
                        out_val.dtype.names is not None
                        and "real" in out_val.dtype.names
                        and "imag" in out_val.dtype.names
                    ):
                        out_val = out_val["real"] + 1j * out_val["imag"]
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
    if func_name in ["tdtf", "costfunlsq"]:
        python_out = func(tfun_test, *args)
    else:
        python_out = func(*args)
    # Ignore second output from Python version of thzgen
    if func_name in ["thzgen", "tdnoisefit"]:
        python_out = python_out[0]
    if func_name == "tdnoisefit":
        # Set absolute tolerance equal to 2 * epsilon for the array dtype
        np.testing.assert_allclose(matlab_out[0], python_out, atol=1e-4)
    elif func_name == "tdnll":
        np.testing.assert_allclose(
            matlab_out[0], python_out[0],
            atol=2 * np.finfo(python_out[0].dtype).eps
        )
        np.testing.assert_allclose(
            matlab_out[1], python_out[1],
            atol=2 * np.finfo(python_out[1].dtype).eps
        )
    else:
        np.testing.assert_allclose(
            matlab_out[0], python_out, atol=2 * np.finfo(python_out.dtype).eps
        )
