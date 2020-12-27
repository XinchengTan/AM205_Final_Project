import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx

from evals import standardize_square_mat


def plot_matrix(mat, title="", cmap="BuPu", figsize=(10, 7)):
  fig, ax = plt.subplots(figsize=figsize)
  if not title:
    title = "Matrix of %d neurons" % mat.shape[0]
  ax.set_title(title, fontsize=16)
  sns.heatmap(mat, cmap=cmap)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  plt.show()


# Plot precision matrix
def plot_prec(prec, alpha, ax=None, standardize=True, label="", cmap="viridis"):
  P = np.array(prec)
  if standardize:
    P = standardize_square_mat(prec)
  if ax:
    sns.heatmap(P, cmap=cmap, ax=ax)
  else:
    ax = sns.heatmap(P, cmap=cmap)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  ax.set_title(r"Precision Matrix [%s, $\lambda$ = %.2f]" % (label, alpha))
  plt.show()


# Plot adjacency matrix
def plot_adj_mat(A, ax=None, label="", include_negs=False):
  plt.figure(figsize=(12, 9))
  A2 = A * 3000
  cmap = "cividis" if not include_negs else "bwr"
  if ax:
    sns.heatmap(A2, cmap=cmap, ax=ax)
  else:
    ax = sns.heatmap(A2, cmap=cmap)
  ax.set_xlabel("Neurons", fontsize=18)
  ax.set_ylabel("Neurons", fontsize=18)
  total_edges = len(A[A != 0])
  if not include_negs:
    ax.set_title("Adjacency Matrix [%s] [%d edges]" % (label, total_edges), fontsize=20)
  else:
    pos_edges = len(A[A > 0])
    neg_edges = len(A[A < 0])
    ax.set_title("Adjacency Matrix [%s] [%d edges: %d+, %d-]" % (label, total_edges, pos_edges, neg_edges))
  plt.show()


# Plot 3 precision matrices
def plot_3_precs(p0, p1, p2, r0, r1, r2, ax=None, standardize_prec=True):

  plot_prec(p0, r0, standardize=standardize_prec,
            ax=ax, label="Full Sample")

  plot_prec(p1, r1, standardize=standardize_prec,
            ax=ax, label="RLA Uniform")

  plot_prec(p2, r2, standardize=standardize_prec,
            ax=ax, label="RLA Minvar")


def plot_3_adj_mats(A0, A1, A2, ax=None, include_negs=False):

  plot_adj_mat(A0, ax, label="Full Sample", include_negs=include_negs)
  plot_adj_mat(A1, ax, label="RLA Uniform", include_negs=include_negs)
  plot_adj_mat(A2, ax, label="RLA Minvar", include_negs=include_negs)


def plot_metric_over_cs_all_sessions(cs, metric_all, metric_name, session_ids, neuron_cnts=None, save_name=None):
  plt.figure(figsize=(12, 9))
  metric_uni = [[] for _ in cs]
  metric_minvar = [[] for _ in cs]
  for sidx in session_ids:
    for ci in range(len(cs)):
      c = cs[ci]
      metric_uni[ci].extend(metric_all[sidx]["uni"][c])
      metric_minvar[ci].extend(metric_all[sidx]["minvar"][c])
  metric_uni = [np.mean(f) for f in metric_uni]
  metric_minvar = [np.mean(f) for f in metric_minvar]

  plt.plot(cs, metric_uni, "o-", label="RLA uniform")
  plt.plot(cs, metric_minvar, "o-", label="RLA minvar")
  #plt.title("%s [%d neurons]" % (metric_name, neuron_cnts), fontsize=20)
  plt.xlabel("RLA Sampling Percentage", fontsize=18)
  plt.ylabel(metric_name, fontsize=18)
  plt.legend(prop={'size': 20})

  if save_name:
    plt.savefig(save_name)
  plt.show()


def plot_metric_uni_over_cs_all_sessions(cs, metric_all, metric_name, session_ids, neuron_cnts=None, save_name=None):
  # assert cs is sorted ascendingly
  plt.figure(figsize=(12, 9))
  for sidx in session_ids:
    metric = metric_all[sidx]
    mean_metric_uni = [np.mean(metric["uni"][c]) for c in cs]
    plt.plot(cs, mean_metric_uni, "o-", label="RLA uniform [Session %d]" % sidx)

  #plt.title("Uniform RLA Sampling %s [%d neurons]" % (metric_name, neuron_cnts), fontsize=20)
  plt.xlabel("RLA Sampling Percentage", fontsize=18)
  plt.ylabel(metric_name, fontsize=18)
  plt.legend(prop={'size': 13})

  if save_name:
    plt.savefig(save_name)
  plt.show()


def plot_metric_minvar_over_cs_all_sessions(cs, metric_all, metric_name, session_ids, neuron_cnts=None, save_name=None):
  # assert cs is sorted ascendingly
  plt.figure(figsize=(12, 9))
  for sidx in session_ids:
    metric = metric_all[sidx]
    mean_frobs_minvar = [np.mean(metric["minvar"][c]) for c in cs]
    plt.plot(cs, mean_frobs_minvar, "o-", label="RLA min-var [Session %d]" % sidx)

  #plt.title("Min-var RLA Sampling %s [%d neurons]" % (metric_name, neuron_cnts), fontsize=20)
  plt.xlabel("RLA Sampling Percentage", fontsize=18)
  plt.ylabel(metric_name, fontsize=18)
  plt.legend(prop={'size': 13})

  if save_name:
    plt.savefig(save_name)
  plt.show()


# Plot matrix correlation over different c
def plot_corrs_over_cs(cs, corrs, neuron_cnts):
  plt.figure(figsize=(12, 9))
  plt.plot(cs, corrs["uni"], "o-", label="RLA uniform")
  plt.plot(cs, corrs["minvar"], "o-", label="RLA minvar")
  plt.title("Precision Matrix Correlation [n = %d neurons]" % neuron_cnts, fontsize=20)
  plt.xlabel("RLA Sample Size", fontsize=18)
  plt.ylabel(r"Matrix Correlation", fontsize=18)
  plt.legend(prop={'size': 13})

  plt.show()


# Plot edge estimation accuracy over different c
def plot_acc_over_cs(cs, accs, neuron_cnts):
  plt.figure(figsize=(12, 9))
  plt.plot(cs, accs["uni"], "o-", label="RLA uniform")
  plt.plot(cs, accs["minvar"], "o-", label="RLA minvar")
  plt.title("Graph Structure Accuracy [n = %d neurons]" % neuron_cnts, fontsize=20)
  plt.xlabel("RLA Sample Size", fontsize=18)
  plt.ylabel(r"Graph Structure Accuracy", fontsize=18)
  plt.legend(prop={'size': 13})
  plt.show()


# Plot TPR over different c
def plot_tpr_over_cs(cs, tprs, neuron_cnts):
  plt.figure(figsize=(12, 9))
  plt.plot(cs, tprs["uni"], "o-", label="RLA uniform")
  plt.plot(cs, tprs["minvar"], "o-", label="RLA minvar")
  plt.title("True Positive Rate of Edge Estimation [n = %d neurons]" % neuron_cnts, fontsize=20)
  plt.xlabel("RLA Sample Size", fontsize=18)
  plt.ylabel(r"TPR", fontsize=18)
  plt.legend(prop={'size': 13})
  plt.show()


# Plot FDR over different c
def plot_fdr_over_cs(cs, fdrs, neuron_cnts):
  plt.figure(figsize=(12, 9))
  plt.plot(cs, fdrs["uni"], "o-", label="RLA uniform")
  plt.plot(cs, fdrs["minvar"], "o-", label="RLA minvar")
  plt.title("False Discovery Rate of Edge Estimation [n = %d neurons]" % neuron_cnts, fontsize=20)
  plt.xlabel("RLA Sample Size", fontsize=18)
  plt.ylabel(r"FDR", fontsize=18)
  plt.legend(prop={'size': 13})
  plt.show()


# Plot connectivity graph based on an adjacency matrix and a location matrix
def plot_connectivity_graph(A, xys, cmap=plt.cm.coolwarm, ax=None, label=""):
  plt.figure(figsize=(12, 9))

  G = nx.convert_matrix.from_numpy_array(A)
  colors = [A[i][j] for i, j in G.edges]
  nx.draw_networkx(G, pos=xys, node_color="orange", alpha=0.85,
                   width=2, edge_cmap=cmap, edge_color=colors)
  plt.axis('equal')
