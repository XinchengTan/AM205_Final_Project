# Functions for applying the GM and its evaluation metrics
import numpy as np
from scipy.stats import pearsonr

from sklearn.covariance import GraphicalLassoCV, graphical_lasso

from utils import check_PD
from RLA import RLA_cov


def est_fc_3covs(dataX, c, alpha, assumed_centered=False):
  if not assumed_centered:
    mu = np.mean(dataX, axis=1)
    dataX = np.transpose(np.transpose(dataX) - mu)

  sample_cov = standardize_square_mat(np.cov(dataX, bias=False))
  check_PD(sample_cov, "Full Sample Covariance")

  rla_uni_cov = RLA_cov(dataX, c=c, standardize=True, sampler="uniform")
  check_PD(rla_uni_cov, "RLA-uniform Covariance")

  rla_minvar_cov = RLA_cov(dataX, c=c, standardize=True, sampler="minvar")
  check_PD(rla_minvar_cov, "RLA-minvar Covariance")

  # Apply graphical model
  gmbase_cov, gmbase_prec = graphical_lasso(sample_cov, alpha=alpha)
  gmuni_cov, gmuni_prec = graphical_lasso(rla_uni_cov, alpha=alpha)
  gmminvar_cov, gmminvar_prec = graphical_lasso(rla_minvar_cov, alpha=alpha)

  return sample_cov, rla_uni_cov, rla_minvar_cov, gmbase_prec, gmuni_prec, gmminvar_prec


# Build connectivity graph from inverse covariance matrix
def getGraph_from_prec(prec, standardize=True, tol=None, qt=0.05):
  """

  :param prec: Estimated precision matrix
  :param standardize: True if prec needs to be standardized
  :param tol: absolute value of the threshold for to designate an edge
  :param qt: quantile cutoff for edge detection
  :return:
  """
  std_prec = standardize_square_mat(prec) if standardize else prec
  A = np.zeros_like(std_prec)
  P = np.abs(std_prec)
  if tol is None:
    tol = np.quantile(P[P != 0], qt)  # prec is flattened in the computation
  A[P > tol] = 1.0
  # print("All prec > tol?", np.all(prec > tol))
  # print("Min prec:", np.min(prec), "Max prec:", np.max(prec))
  return A


# Computes 3 adjacency matrices based on the 3 GM-estimated precision matrices
def get_3_graphs(p0, p1, p2, std_prec=True, tol=None, include_negs=False):
  A0 = getGraph_from_prec(p0, std_prec, tol=tol)
  A1 = getGraph_from_prec(p1, std_prec, tol=tol)
  A2 = getGraph_from_prec(p2, std_prec, tol=tol)
  return A0, A1, A2


# Two numerical evaluation metrics on the GM-estimated precision matrices
def compare_precs(baseP, uniP, minvarP, verbose=False):
  # Frobenius norm
  fro_uni = frob_norm(baseP, uniP)
  fro_minvar = frob_norm(baseP, minvarP)

  # Matrix correlation
  corr_uni = mat_corr(baseP, uniP)
  corr_minvar = mat_corr(baseP, minvarP)

  if verbose:
    print("Frobenius(base, RLA-uni): ", fro_uni)
    print("Correlation(base, RLA-uni): ", corr_uni)
    print("Frobenius(base, RLA-minvar): ", fro_minvar)
    print("Correlation(base, RLA-minvar): ", corr_minvar)

  return fro_uni, fro_minvar, corr_uni, corr_minvar


# Compare the three adjacency matrices
def compare_graphs(baseA, uniA, minvarA, include_negs=False, verbose=True):
  match_pct_uni = matching_pct(baseA, uniA)
  match_pct_minvar = matching_pct(baseA, uniA)
  if verbose:
    print("Recovered edge percentage of Baseline by RLA-uniform: %.2f" % match_pct_uni)
    print("Recovered edge percentage of Baseline by RLA-minvar: %.2f" % match_pct_minvar)

  if not include_negs:
    tpr_uni, fdr_uni = binary_edge_tpr_fdr(baseA, uniA)
    tpr_minvar, fdr_minvar = binary_edge_tpr_fdr(baseA, minvarA)
    if verbose:
      print("RLA-uniform TPR: %.2f; FDR: %.2f" % (tpr_uni, fdr_uni))
      print("RLA-minvar TPR: %.2f; FDR: %.2f" % (tpr_minvar, fdr_minvar))

    return match_pct_uni, match_pct_minvar, tpr_uni, tpr_minvar, fdr_uni, fdr_minvar
  else:
    print("We don't support ternary eval yet")


# Frobenius norm of 2 numerical matrices
def frob_norm(mat1, mat2):
  assert np.shape(mat1) == np.shape(mat2), "Input matrices have different dimensions!"
  m, n = np.shape(mat1)
  diff = mat1 - mat2
  fro_n = np.linalg.norm(diff, ord="fro")
  return fro_n / (m * n)


# Pearson's correlation of 2 numerical matrices (the diagonal entries excluded)
def mat_corr(A, B, square_symmetric=True):
  shapeA, shapeB = np.shape(A), np.shape(B)
  assert shapeA == shapeB, "Input matrices has different shapes: %s, %s" % (str(shapeA), str(shapeB))
  p = shapeA[0]
  if square_symmetric:
    idxs = np.triu_indices(p, 1) # Exclude diagonal
    upperA, upperB = A[idxs], B[idxs]
    return pearsonr(upperA, upperB)[0]
  return pearsonr(np.matrix(A).flatten(), np.matrix(B).flatten())[0]


# Matching percentage of two adjacency matrices (the diagonal entries excluded)
def matching_pct(A1, A2):
  # Exclude diagonal entries
  assert np.shape(A1) == np.shape(A2), "Input matrices must have the same shape!"
  p = np.shape(A1)[0]

  upper1, upper2 = A1[np.triu_indices(p, 1)], A2[np.triu_indices(p, 1)]
  diff = np.abs(upper1 - upper2)
  return 1.0 - np.mean(diff)


# Accuracy of two estimated graph structure
def binary_edge_tpr_fdr(trueA, A):
  """
  :param trueA: Baseline adjacency matrix
  :param A: Actual adjacency matrix
  :return: TPR and FDR or the non-diagonal entries
  """
  assert np.shape(trueA) == np.shape(A), "Dimension does not match!"
  p, n = trueA.shape
  assert p == n, "Input matrix is not square!"

  upp1, upp2 = trueA[np.triu_indices(p, 1)], A[np.triu_indices(p, 1)]
  diff = upp1 - upp2

  true_edges = len(upp1[upp1 == 1])
  tpr = len(diff[(diff == 0) & (upp1 == 1)]) / true_edges  # edge matches

  est_edges = len(upp2[upp2 == 1])
  #print(est_edges, true_edges)

  if est_edges > 0:
    fdr = len(diff[diff == -1]) / est_edges  # false discovery rate
  else:
    fdr = None
    print("No edge detected in A!")
  return tpr, fdr


def standardize_square_mat(mat):
  p, n = np.shape(mat)
  assert p == n, "Input matrix is not a square matrix!"

  sqrt_diag = np.reshape(np.diag(mat) ** 0.5, (p, 1))
  standardizer = sqrt_diag * sqrt_diag.T
  std_prec = mat / standardizer
  return std_prec

