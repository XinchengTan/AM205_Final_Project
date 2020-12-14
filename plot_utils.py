import matplotlib.pyplot as plt
import seaborn as sns


def plot_matrix(mat, title="", cmap="BuPu", figsize=(10, 7)):
  fig, ax = plt.subplots(figsize=figsize)
  if not title:
    title = "Matrix of %d neurons" % mat.shape[0]
  ax.set_title(title, fontsize=16)
  sns.heatmap(mat, cmap=cmap)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  plt.show()
