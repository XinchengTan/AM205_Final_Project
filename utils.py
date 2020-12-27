# Helper functions
from globals import *
from NearestPSD import shrinking

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.covariance import EmpiricalCovariance


def check_PD(mat, mat_name=""):
  if mat.shape[0] != mat.shape[1]:
    raise ValueError("Input matrix for PD check must be a square matrix!")

  nan_entries = np.where(np.isnan(mat))
  inf_entries = np.where(np.isinf(mat))
  if len(nan_entries[0]) > 0:
    print("There are %d nan entries in mat!" % len(nan_entries[0]))
  if len(inf_entries[0]) > 0:
    print("There are %d inf entries in mat!" % len(inf_entries[0]))

  is_pd = np.all(np.linalg.eigvals(mat) > 0)
  chol_is_pd = shrinking.checkPD(mat)
  if is_pd and chol_is_pd:
    return True
  elif is_pd and not chol_is_pd:
    print("Checker inconsistent with Cholesky decomposition checker!")
    return None
  else:
    is_psd = np.all(np.linalg.eigvals(mat) >= 0)
    if not is_psd:
      print("%s Matrix is not positive semi-definite!" % mat_name)
  return False


def nan_percentage(mat):
  nan_cnts = len(np.where(np.isnan(mat))[0])
  return nan_cnts / np.size(mat)


# Tuning curve of neuron responses to tuning variables
def tuning_curve(fsp, var, bins=None, binrange=None, verbose=False):
  assert fsp.shape[1] == len(var), "Time axis of fsp and tuning variable does not align!"
  assert np.ndim(var) == 1 or (np.ndim(var) == 2 and np.shape(var)[1] == 1), "Tuning variable should be 1-dimensional!"

  n = fsp.shape[0]
  var = np.reshape(var, len(var))

  if bins is None:
    # var is already categorical
    var_vals = sorted(list(set(var)))
    avg_tuning_curve = np.zeros((len(var_vals), n))
    for vi, ival in enumerate(var_vals):
      fsp_ival = fsp[:, var == ival]
      avg_tuning_curve[vi] = np.nanmean(fsp_ival, axis=1)
    return var_vals, avg_tuning_curve

  else:
    # If var is continuous, bin the values into evenly-spaced groups
    var_bins = np.linspace(binrange[0], binrange[1], bins)
    bins_cnt = len(var_bins) - 1
    avg_tuning_curve = np.zeros((len(var_bins), n))
    empty_bin = set()
    for i in range(bins_cnt):
      fsp_ibin = fsp[:, (var >= var_bins[i]) & (var < var_bins[i + 1])]
      if fsp_ibin.shape[1] > 0:
        avg_tuning_curve[i] = np.nanmean(fsp_ibin, axis=1)
      else:
        # Discard the bin if it has never occurred
        empty_bin.add(i)
    var_bins = np.array([var_bins[i] for i in range(bins_cnt) if i not in empty_bin])
    avg_tuning_curve = np.array([avg_tuning_curve[i, :] for i in range(bins_cnt) if i not in empty_bin])
    return var_bins, avg_tuning_curve


# Correlation matrix of the tuning curve matrix across a neuron population
def tc_mat_corr(tc_mat, checkNan=True):
  # tc_mat has shape: (tuning_vars, neurons)
  tc_sim = np.array(pd.DataFrame(tc_mat).corr())  # pd.corr ignores nan

  if checkNan and np.any(np.isnan(tc_sim)):
    print("Tuning Curve Similarity Matrix contains NaN!")
  return tc_sim


# Tuning curve correlation of two neurons
def tc_corr(tc1, tc2):
  return pearsonr(tc1, tc2)


def isSymmetric(X):
  return np.allclose(X, X.T, rtol=1e-8, atol=1e-8)


def isPSD(X):
  if X.ndim > 2:  return False
  if not isSymmetric(X):
    print("Warning: Input matrix is not symmetric!")
  eigenvals = np.linalg.eigvals(X)
  return all(eigenvals >= 0)


def isPD(X):
  if np.ndim(X) > 2: return False
  if not isSymmetric(X):
    print("Warning: Input matrix is not symmetric!")
  eigenvals = np.linalg.eigvals(X)
  return all(eigenvals > 0)


def newtons_PDCorrection(M0, *, M1=None, fbs=None, tol=10**(-6), maxIterations=None, checkM0=True):
  alpha = shrinking.newton(M0, M1=M1, fbs=fbs, tol=tol, maxIterations=maxIterations, checkM0=checkM0)
  pd_matrix = alpha * M1 + (1 - alpha) * M0
  return pd_matrix


