# Computes similarity of two matrices

import numpy as np
from scipy.stats import pearsonr

from sklearn.covariance import GraphicalLassoCV, graphical_lasso

# TODO: If time permits, simulate a multivariate Gaussian distribution data and compare performance?

def est_connectivity(X, gm="Glasso", assume_centered=False):
  if gm == "Glasso":  # Default: Glasso
    glasso = GraphicalLassoCV(assume_centered=assume_centered, cv=5).fit(X)
    return glasso.covariance_, glasso.get_precision(), glasso.alpha_


# Frobenius norm of 2 numerical matrices
def frob_norm(mat1, mat2):
  assert np.shape(mat1) == np.shape(mat2), "Input matrices have different dimensions!"
  m, n = np.shape(mat1)
  diff = mat1 - mat2
  fro_n = np.linalg.norm(diff, ord="fro")
  return fro_n / (m * n)


# correlation of 2 numerical matrices
def mat_corr(A, B, square_symmetric=True):
  shapeA, shapeB = np.shape(A), np.shape(B)
  assert shapeA == shapeB, "Input matrices has different shapes: %s, %s" % (str(shapeA), str(shapeB))
  p = shapeA[0]
  if square_symmetric:
    idxs = np.triu_indices(p, 1) # Exclude diagonal
    upperA, upperB = A[idxs], B[idxs]
    return pearsonr(upperA, upperB)[0]
  return pearsonr(np.matrix(A).flatten(), np.matrix(B).flatten())[0]


# Matching percentage
def matching_pct(A1, A2):
  # Exclude diagonal entries
  assert np.shape(A1) == np.shape(A2), "Input matrices must have the same shape!"
  p = np.shape(A1)[0]

  upper1, upper2 = A1[np.triu_indices(p, 1)], A2[np.triu_indices(p, 1)]
  diff = np.abs(upper1 - upper2)
  return 1.0 - np.mean(diff)


# accuracy of two estimated graph structure
def binary_edge_tpr_fdr(trueA, A):
  """

  :param trueA:
  :param A:
  :param binary:
  :return:
  """
  assert np.shape(trueA) == np.shape(A), "Dimension does not match!"
  p, n = trueA.shape
  assert p == n, "Input matrix is not square!"

  upp1, upp2 = trueA[np.triu_indices(p, 1)], trueA[np.triu_indices(p, 1)]
  diff = upp1 - upp2
  true_edges = len(upp1[upp1 == 1])
  tpr = 1.0 - len(diff[diff == 1]) / true_edges  # edge matches
  fdr = 1.0 - len(diff[diff == -1]) / true_edges  # false discovery
  return tpr, fdr



def ternary_edge_tpr_fdr(trueA, A):
  """
  Returns the percentage of matched edges, matched positive edges and matched negative edges
  :param trueA:
  :param A:
  :return:
  """
  # Exclude diagonal entries
  assert np.shape(trueA) == np.shape(A), "Input matrices must have the same shape!"
  p, n = trueA.shape
  assert p == n, "Input matrix is not square!"

  upp1, upp2 = trueA[np.triu_indices(p, 1)], A[np.triu_indices(p, 1)]
  pos_p1 = len(upp2[(upp1 == 1) & (upp2 == 1)]) / len(upp1[upp1 == 1])
  pos_p2 = len(upp1[(upp1 == 1) & (upp2 == 1)]) / len(upp2[upp2 == 1])
  neg_p1 = len(upp2[(upp1 == -1) & (upp2 == -1)]) / len(upp1[upp1 == -1])
  neg_p2 = len(upp1[(upp1 == -1) & (upp2 == -1)]) / len(upp2[upp2 == -1])
  return pos_p1, pos_p2, neg_p1, neg_p2


  # assert trueA.shape == A.shape, "Dimension does not match!"
  # m, n = trueA.shape
  # mn = m * n
  # total_matches = len(np.where(trueA == A)[0])
  # pos_matches = len(np.where((trueA == A) & (trueA == 1))[0])
  # neg_matches = len(np.where((trueA == A) & (trueA == -1))[0])
  #
  # pos_acc, neg_acc = pos_matches / mn, neg_matches / mn
  # return pos_acc, neg_acc



def standardize_square_mat(mat):
  p, n = np.shape(mat)
  assert p == n, "Input matrix is not a square matrix!"

  sqrt_diag = np.reshape(np.diag(mat) ** 0.5, (p, 1))
  standardizer = sqrt_diag * sqrt_diag.T
  std_prec = mat / standardizer
  return std_prec


# Build connectivity graph from inverse covariance matrix
def getGraph_from_prec(prec, standardize=True, tol=None, qt=0.05, include_negs=False):
  """

  :param prec: Estimated precision matrix
  :param standardize:
  :param tol:
  :param qt:
  :param include_negs:
  :return:
  """
  std_prec = standardize_square_mat(prec) if standardize else prec
  A = np.zeros_like(std_prec)
  P = np.abs(std_prec)
  if tol is None:
    tol = np.quantile(P[P != 0], qt)  # prec is flattened in the computation
  if include_negs:
    A[std_prec > tol] = 1.0
    A[std_prec < -tol] = -1.0
  else:
    A[P > tol] = 1.0
  # print("All prec > tol?", np.all(prec > tol))
  # print("Min prec:", np.min(prec), "Max prec:", np.max(prec))
  return A







