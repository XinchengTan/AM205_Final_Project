# Loads .mat datasets into RecordingData objects

from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.spatial.distance import pdist, squareform
from sklearn.covariance import EmpiricalCovariance
import seaborn as sns

import globals
from NearestPSD import shrinking
import utils


# Recording with 32 drifting gratings
# Keys:
#  - Fsp: a matrix of size number of neurons by number of timepoints of deconvolved fluorescence traces.
#  - stat: single cell statistics of cell detection algorithm (Suite2p)
#  - stim: properties of the stimulus
#  - med: estimated 3D position of cells in tissue.
# TODO: Think about building an 'extensible' dataset, to maximize the container's ability to accept data of other formats.
# TODO: 1. Create dicts/NamedTuples for neuron-based and for time-based auxiliary information.
class RecordingData(object):
  """
  This class is a container for calcium imaging data with auxiliary physiological and behavioral information.

  The cellular fluorescence traces in all layers of the brain are recorded simultaneously.
  Dataset with 10 layers are scanned at a rate of 3 Hz, those with 12 layers 2.5 Hz.
  """
  def __init__(self, recording_data=None, data_fp=None):
    if data_fp is not None:
      self.recording = scipy.io.loadmat(data_fp)

      # Neural signals
      self.fsp = np.array(self.recording.get("Fsp")[:, 1:])  # Neural responses * timepoints (remove t=0 with all zeros)
      self.neuron_counts = self.fsp.shape[0]
      self.timestamp_counts = self.fsp.shape[1]  # TODO: Is the timestamp evenly spaced?
      self.timestamp_offset = 0

      # Estimated 3D position of the cell
      self.neuron_xyz = np.array(self.recording.get("med"))  # Neural responses * 3
      self.layers_zcoord = sorted(set(self.neuron_xyz[:, 2]))
      self.layer_counts = len(self.layers_zcoord)
      # Sort neuron index by layer
      self.layer2neurons = {zcoord: np.where(self.neuron_xyz[:, 2] == zcoord)[0] for zcoord in self.layers_zcoord}
      self.single_cell_stat = np.array(self.recording.get("stat")[0])  # [‘npix’]?

    else:
      # Copy the attributes from another RecordingData object
      # TODO: Jupyter notebook's autoreload cannot handle this check, disable it for now
      # assert isinstance(recording_data, RecordingData)
      self.__dict__ = deepcopy(recording_data.__dict__)

  def validate(self):
    # Validate the initialization
    pass

  def get_layers_fsp(self, layers=None):
    if layers is None:
      layers = list(range(self.layer_counts))
    fsp = []
    for layer in layers:
      fsp.extend(self.get_layer_fsp(layer))
    return np.array(fsp)

  def get_layer_fsp(self, layer):
    if not (0 <= layer < self.layer_counts):
      print("[RecordingData] Invalid layer! There are %d neuron layers of this dataset!" % self.layer_counts)
    zcoord = self.layers_zcoord[layer]
    return np.array(self.fsp[self.layer2neurons[zcoord]])

  def get_layers_xyz(self, layers=None):
    if layers is None:
      layers = list(range(self.layer_counts))
    new_xyz = []
    for layer in layers:
      xyz = self.get_layer_xyz(layer)
      new_xyz.extend(xyz)
    return np.array(new_xyz)

  def get_layer_xyz(self, layer):
    if not (0 <= layer < self.layer_counts):
      print("[RecordingData] Invalid layer! There are %d neuron layers of this dataset!" % self.layer_counts)
    zcoord = self.layers_zcoord[layer]
    res = np.array(self.neuron_xyz[np.where(self.neuron_xyz[:, 2] == zcoord)[0]])
    return res

  # Visualizations
  def display_field_shapes(self):
    print("neuron counts:", self.neuron_counts)
    print("layer counts:", self.layer_counts)
    for layer in range(self.layer_counts):
      print("Layer %d: %d neurons" % (layer, len(self.layer2neurons[self.layers_zcoord[layer]])))
    print("timestamp counts", self.timestamp_counts)

  def display_aggregated_stats(self):
    pass

  def display_fsp_distribution_all_time(self, bins=50):
    fsp_mean, fsp_std = np.mean(self.fsp, axis=0), np.std(self.fsp, axis=0)

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)
    ax1.hist(fsp_mean, bins, color="green", alpha=0.8)
    ax1.set_title("Distribution of the Population Mean of the Calcium Imaging Signal")
    ax1.set_xlabel("Deconvolved Fluorescence Trace")
    ax1.set_ylabel("Number of Timepoints")

    ax2 = fig.add_subplot(122)
    ax2.hist(fsp_std, bins, color="green", alpha=0.6)
    ax2.set_title("Distribution of the Population Std of the Calcium Imaging Signal")
    ax2.set_xlabel("Deconvolved Fluorescence Trace")
    ax2.set_ylabel("Number of Neurons")
    plt.show()

  def display_fsp_distribution_instant(self, timepoint=0, xrange=(0, 200), bins=100):
    assert timepoint < self.timestamp_counts, "Only %d timepoints are available in this recording!" % self.timestamp_counts

    instant_fsp = self.fsp[:, timepoint]
    plt.hist(instant_fsp, bins, color="blue", alpha=0.7)
    plt.xlim(xmin=xrange[0], xmax=xrange[1])
    plt.title("Snapshot Distribution of the Calcium Imaging Signal")
    plt.xlabel("Deconvolved Fluorescence Trace")
    plt.ylabel("Number of Neurons")
    plt.show()

  def display_rand_subset(self, size=5, use_seed=False):
    if use_seed: np.random.seed(globals.SEED) # TODO: Edit me!
    subset_fsp = self.fsp[np.random.choice(range(self.neuron_counts), size, replace=False)]
    for i in range(size):
      plt.plot(subset_fsp[i])
      title = "Calcium Imaging Value of Random Neuron %d" % i
      if isinstance(self, SpontaneousRecordingData):
        title += " [Session: %s]" % self.session_name
      plt.title(title, fontsize=16, fontweight='bold')
      plt.xlabel("Timestamp")
      plt.ylabel("Calcium Imaging Value")
      plt.show()

  def display_neuron_3d(self, session_name="", layers=None):
    ax = plt.axes(projection='3d')
    if layers is None:
      ax.scatter3D(self.neuron_xyz[:, 0], self.neuron_xyz[:, 1], self.neuron_xyz[:, 2],
                   c=self.neuron_xyz[:, 2], cmap="viridis", linewidth=0.5, s=18)
    else:
      zs = [self.layers_zcoord[layer] for layer in layers]
      neurons = [self.layer2neurons[z] for z in zs]
      for ns in neurons:
        ax.scatter3D(self.neuron_xyz[ns, 0], self.neuron_xyz[ns, 1], self.neuron_xyz[ns, 2],
                     c=self.neuron_xyz[ns, 2], cmap="viridis", linewidth=0.5)
    if isinstance(self, SpontaneousRecordingData):
      ax.set_title("Neuron Locations [Session: %s]" % self.session_name, fontsize=16, fontweight="bold")
    else:
      ax.set_title("Neuron Locations [Session: %s]" % session_name, fontsize=16, fontweight="bold")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_zlabel("Depth (um)")


  # Auxiliary Information
  def inter_neuron_distance(self, type=globals.EUCLIDEAN, plot=False):
    # Inter-neuron distance, default measure being Euclidean
    dist_condensed = pdist(self.neuron_xyz)
    if plot:
      # Plot the distribution of the inter-neuron distance
      plt.hist(dist_condensed, bins=30, color="grey")
      plt.title("Distribution of inter-neuron distance")
      plt.xlabel("Distance (um)")
      plt.ylabel("Count of Neuron Pairs")
      plt.show()
    return squareform(dist_condensed)

  def sample_covariance_matrix(self, corr=False, neuron_range=None, layer=None, check_psd=False, display=False, cmap="BuPu"):

    X, layer_name = (self.get_layer_fsp(layer), str(layer)) if layer is not None else (self.fsp, "all")
    X = X[neuron_range[0]:neuron_range[1]] if neuron_range else X
    cov, mat_name = (np.cov(X), "Covariance") if not corr else (np.corrcoef(X), "Correlation")

    if check_psd:
      is_pd = np.all(np.linalg.eigvals(cov) > 0)
      chol_is_pd = shrinking.checkPD(cov)
      if is_pd and chol_is_pd:
        print("Sample %s Matrix is positive definite!" % mat_name)
      elif is_pd and not chol_is_pd:
        print("Checker inconsistent with Cholesky decomposition checker!")
      else:
        is_psd = np.all(np.linalg.eigvals(cov) >= 0)
        print("Sample %s Matrix is positive semi-definite: %s" % (mat_name, is_psd))

    if display:
      fig, ax = plt.subplots(figsize=(10, 7))
      title = "Sample %s Matrix - Layer: %s" % (mat_name, layer_name)
      if isinstance(self, SpontaneousRecordingData):
        title += " [Session: %s]" % self.session_name
      ax.set_title(title)
      sns.heatmap(cov, cmap=cmap)
      ax.set_xlabel("Neurons")
      ax.set_ylabel("Neurons")
      plt.show()
    return cov

  def sample_precision_matrix(self):
    emp_cov_est = EmpiricalCovariance().fit(self.fsp.transpose())
    prec = emp_cov_est.precision_
    title = "Empirical Correlation"
    if isinstance(self, SpontaneousRecordingData):
      title += " [Session: %s]" % self.session_name
    plt.title(title, fontsize=16, fontweight="bold")
    plt.imshow(prec, cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    return prec


# Spontaneous Recording
class SpontaneousRecordingData(RecordingData):

  def __init__(self, recording_data=None, data_fp=None):
    super().__init__(recording_data, data_fp)
    if data_fp is not None:
      # Behavior data
      beh_data = self.recording.get("beh")
      self.run_speed = np.array(beh_data["runSpeed"][0][0][1:])
      self.motion_svd = np.array(beh_data[0]["face"][0]["motionSVD"][0][0][1:])
      self.motion_mask = np.array(beh_data[0]["face"][0]["motionMask"][0][0])
      self.avgframe = np.array(beh_data[0]["face"][0]["avgframe"][0][0])  # avg frame of face ROI
      self.pupil_area = np.array(beh_data[0]["pupil"][0]["area"][0][0][1:])
      self.pupil_com = np.array(beh_data[0]["pupil"][0]["com"][0][0][1:])  # pupil center of mass
      self.eye_motion_svd = np.array(beh_data[0]["eye"][0]["motionSVD"][0][0][1:])
      self.whisker_motion_svd = np.array(beh_data[0]["whisker"][0]["motionSVD"][0][0][1:])
      self.whisker_motion_mask = np.array(beh_data[0]["whisker"][0]["motionMask"][0][0])
      self.face_roi_shape = self.avgframe.shape

      # TODO: 1. What does the list of indices corresponding to "spont_black" / "spont_grey" point to?
      self.metadata = self.recording.get("db")[0][0]
      self.session_name = str(self.metadata[globals.SESSION_NAME][0])
    else:
      # Copy the attributes from another RecordingData object
      # TODO: Jupyter notebook's autoreload cannot handle this check, disable it for now
      # assert isinstance(recording_data, RecordingData)
      self.__dict__ = deepcopy(recording_data.__dict__)

  def display_field_shapes(self):
    super().display_field_shapes()
    print("running speed", self.run_speed.shape)
    print("face motion mask", self.motion_mask.shape)
    print("face motion svd", self.motion_svd.shape)
    print("avg frame", self.avgframe.shape)
    print("pupil area", self.pupil_area.shape)
    print("pupil com", self.pupil_com.shape)
    print("eye motion svd", self.eye_motion_svd.shape)
    print("whisker motion mask", self.whisker_motion_mask.shape)
    print("whisker motion SVD", self.whisker_motion_svd.shape)

  def display_run_speed(self):
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)
    ax1.plot(self.run_speed, color="orange")
    ax1.set_title("Running Speed [Session: %s]" % self.session_name, fontsize=15, fontweight='bold')
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Running Speed [mm/0.4sec]")

    ax2 = fig.add_subplot(122)
    ax2.hist(self.run_speed, bins=100, color="red", alpha=0.5)
    ax2.set_title("Running Speed Distribution", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Running Speed [mm/0.4sec]")
    ax2.set_ylabel("Occurrences")

    plt.show()

  def display_pupil_area(self):
    print("Pupil Area data NaN percentage: ", utils.nan_percentage(self.pupil_area))
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)
    ax1.plot(self.pupil_area, color="grey")
    ax1.set_title("Pupil Area [Session: %s]" % self.session_name, fontsize=15, fontweight='bold')
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Pupil Area")

    ax2 = fig.add_subplot(122)
    ax2.hist(self.pupil_area, bins=100, color="grey", alpha=0.75)
    ax2.set_title("Pupil Area Distribution", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Pupil Area")
    ax2.set_ylabel("Occurrences")

    plt.show()

  def display_aggregated_stats(self):
    fsp_mean, fsp_std = np.nanmean(self.fsp, axis=0), np.nanstd(self.fsp, axis=0)
    run_speed_mean, run_speed_std = np.nanmean(self.run_speed), np.nanstd(self.run_speed)
    pa_mean, pa_std = np.nanmean(self.pupil_area), np.nanstd(self.pupil_area)
    print("fsp:\n- mean:", fsp_mean)
    print("- std:", fsp_std)
    print("running speed:\n-mean:", run_speed_mean)
    print("- std:", run_speed_std)
    print("- range: %.2f ~ %.2f" % (np.nanmin(self.run_speed), np.nanmax(self.run_speed)))
    print("pupil area:\n- mean:", pa_mean)
    print("- std:", pa_std)
    print("- range: %.2f ~ %.2f" % (np.nanmin(self.pupil_area), np.nanmax(self.pupil_area)))

    return {"fsp": {"mean": fsp_mean, "std": fsp_std},
            "run_speed": {"mean": run_speed_mean, "std": run_speed_std},
            "pupil_area": {"mean": pa_mean, "std": pa_std}}




class PartialSpontDataset(SpontaneousRecordingData):

  def __init__(self, dataset: SpontaneousRecordingData, *, layersOI=None,
               start_timepoint=None, end_timepoint=None,
               ul_coord=None, lr_coord=None):
    """
    This class samples a partial recording from an original calcium imaging dataset along 3 possible dimensions:
      - neuron layer
      - x-y plane
      - timepoints (observations)

    :param dataset: File path to the dataset to load the original RecordingData
    :param layersOI: A list of layer indices to include in this PartialRecordingDataset
    :param start_timepoint: (inclusive) start time index with respect to the original full RecordingData
    :param end_timepoint: (exclusive) end time index with respect to the original full RecordingData
    :param ul_coord: upper left x-y coordinate of the neural cube to sample
    :param lr_coord: lower right x-y coordinate of the neural cube to sample
    """
    if type(dataset).__name__ != "SpontaneousRecordingData":
      raise AttributeError("Input dataset has to be of type SpontaneousRecordingData")

    # Initialize all fields with that of the input RecordingData
    super().__init__(recording_data=dataset)
    self.original_dataset = dataset

    # Chunk along layer (z-axis)
    if layersOI:
      # Overwrite the fields containing the neuron data, leaving only those in the 'layers of interest'.
      self.fsp = dataset.get_layers_fsp(layers=layersOI)
      self.neuron_counts = self.fsp.shape[0]
      self.neuron_xyz = dataset.get_layers_xyz(layers=layersOI)
      self.layers_zcoord = sorted(set(self.neuron_xyz[:, 2]))
      self.layer_counts = len(self.layers_zcoord)
      self.layer2neurons = {zcoord: np.where(self.neuron_xyz[:, 2] == zcoord)[0] for zcoord in self.layers_zcoord}

    self.display_neuron_3d()
    # # Chunk along time axis
    # self.timestamp_offset = start_timepoint
    #
    # # Check args
    # if not (0 <= start_timepoint < dataset.timestamp_counts):
    #   raise ValueError("Start timepoint must lie in [0, %d)!" % dataset.timestamp_counts)
    # if not (start_timepoint < end_timepoint <= dataset.timestamp_counts):
    #   raise ValueError("End timepoint must lie in (start_tp=%d, %d]!" % (start_timepoint, dataset.timestamp_counts))
    #
    # # Chunk all temporal-based features
    # start_ts = max(0, start_timepoint)
    # end_ts = min(dataset.timestamp_counts, end_timepoint)
    # self.fsp = dataset.fsp[:, start_ts:end_ts]
    # self.timestamp_counts = end_ts - start_ts
    # self.timestamp_offset += start_timepoint
    #
    # # Chunk behavioral features
    # if isinstance(dataset, SpontaneousRecordingData) or isinstance(self.original_dataset, SpontaneousRecordingData):
    #   self.run_speed = dataset.run_speed[start_ts:end_ts, :]
    #   self.motion_svd = dataset.motion_svd[start_ts:end_ts, :]
    #   self.whisker_motion_svd = dataset.whisker_motion_svd[start_ts:end_ts, :]
    #   self.eye_motion_svd = dataset.eye_motion_svd[start_ts:end_ts, :]
    #   self.pupil_area = dataset.pupil_area[start_ts:end_ts, :]
    #   self.pupil_com = dataset.pupil_com[start_ts:end_ts, :]


    #

  def get_original_trace(self):
    return np.array(self.original_dataset.fsp)

  def get_original_time_period(self):
    return (0, self.original_dataset.timestamp_counts)

  def get_original_neuron_xyz(self):
    return np.array(self.original_dataset.neuron_xyz)






class PartialRecordingDataset(RecordingData):

  def __init__(self, dataset: RecordingData, *, layersOI=None,
               start_timepoint=None, end_timepoint=None,
               ul_coord=None, lr_coord=None):
    """
    This class samples a partial recording from an original calcium imaging dataset along 3 possible dimensions:
      - neuron layer
      - x-y plane
      - timepoints (observations)

    :param dataset: File path to the dataset to load the original RecordingData
    :param layersOI: A list of layer indices to include in this PartialRecordingDataset
    :param start_timepoint: (inclusive) start time index with respect to the original full RecordingData
    :param end_timepoint: (exclusive) end time index with respect to the original full RecordingData
    :param ul_coord: upper left x-y coordinate of the neural cube to sample
    :param lr_coord: lower right x-y coordinate of the neural cube to sample
    """
    super().__init__(recording_data=dataset)
    print("Dataset class: ", type(dataset).__name__)
    if isinstance(dataset, PartialRecordingDataset):
      self.original_dataset = dataset.original_dataset
    elif isinstance(dataset, RecordingData) or isinstance(dataset, SpontaneousRecordingData):
      self.original_dataset = dataset

  # TODO: Add necessary getter methods!
  def get_partial_fsp(self, starttime=0, endtime=np.inf, display=False):
    """
    Returns a partial fsp from 'starttime' to 'endtime' (inclusive) with nan entries.
    :param starttime:
    :param endtime:
    :param display:
    :return:
    """
    return np.array(self.fsp)

  def get_original_indices(self):
    pass

  def get_original_trace(self):
    return np.array(self.original_dataset.fsp)

  def get_original_time_period(self):
    return (0, self.original_dataset.timestamp_counts)

  def get_original_neuron_xyz(self):
    return np.array(self.original_dataset.neuron_xyz)


class PartialLayerRecordingDataset(PartialRecordingDataset):

  def __init__(self, dataset: RecordingData, layersOI):
    """
    This class faithfully stores a subset of an input RecordingData among specified layers.

    :param dataset: A RecordingData object, which can be partial.
    :param layersOI: A list of layer indices with respect to 'dataset' instead of the very original dataset.
    """
    super().__init__(dataset=dataset, layersOI=layersOI)
    print("[partial layer] original dataset:", self.original_dataset)

    # Overwrite the fields containing the neuron data, leaving only those in the 'layers of interest'.
    self.fsp = dataset.get_layers_fsp(layers=layersOI)
    self.neuron_counts = self.fsp.shape[0]
    self.neuron_xyz = dataset.get_layers_xyz(layers=layersOI)
    self.layers_zcoord = sorted(set(self.neuron_xyz[:, 2]))
    self.layer_counts = len(self.layers_zcoord)
    self.layer2neurons = {zcoord: np.where(self.neuron_xyz[:, 2] == zcoord)[0] for zcoord in self.layers_zcoord}


class PartialTimeRecordingDataset(PartialRecordingDataset):

  def __init__(self, dataset: RecordingData, start_timepoint, end_timepoint):
    """
    This class faithfully stores a subset of an input RecordingData during a time period.
    All the time-dependent fields are modified to store data during the specified period only.

    Note: The start and end timepoint should be specified with respect to the input RecordingData,
          instead of the very original dataset.

    :param dataset: A RecordingData object, which can be partial.
    :param start_timepoint: Index of the start timestamp in the time period of interest.
    :param end_timepoint: Index of the ending timestamp in the time period of interest.
    """
    super().__init__(dataset=dataset, start_timepoint=start_timepoint, end_timepoint=end_timepoint)
    self.timestamp_offset = start_timepoint

    # Check args
    if not (0 <= start_timepoint < dataset.timestamp_counts):
      raise ValueError("Start timepoint must lie in [0, %d)!" % dataset.timestamp_counts)
    if not (start_timepoint < end_timepoint <= dataset.timestamp_counts):
      raise ValueError("End timepoint must lie in (start_tp=%d, %d]!" % (start_timepoint, dataset.timestamp_counts))

    # Chunk all temporal-based features
    start_ts = max(0, start_timepoint)
    end_ts = min(dataset.timestamp_counts, end_timepoint)
    self.fsp = dataset.fsp[:, start_ts:end_ts]
    self.timestamp_counts = end_ts - start_ts
    self.timestamp_offset += start_timepoint

    # Chunk behavioral features
    if isinstance(dataset, SpontaneousRecordingData) or isinstance(self.original_dataset, SpontaneousRecordingData):
      self.run_speed = dataset.run_speed[start_ts:end_ts, :]
      self.motion_svd = dataset.motion_svd[start_ts:end_ts, :]
      self.whisker_motion_svd = dataset.whisker_motion_svd[start_ts:end_ts, :]
      self.eye_motion_svd = dataset.eye_motion_svd[start_ts:end_ts, :]
      self.pupil_area = dataset.pupil_area[start_ts:end_ts, :]
      self.pupil_com = dataset.pupil_com[start_ts:end_ts, :]


class PartialSpaceRecordingDataset(PartialRecordingDataset):

  def __init__(self, dataset: RecordingData,
               x_range: Tuple[float, float], y_range: Tuple[float, float], ratio=False):
    """
    This class stores a subset of neurons within a specified rectangular region.

    :param dataset: A RecordingData object, which can be partial
    :param x_range: Range of x coordinate
    :param y_range: Range of y coordinate
    :param ratio: If true, x_range and y_range are interpreted as the proportion w.r.t. dataset's neurons at each layer
    """
    super().__init__(dataset=dataset, ul_coord=x_range, lr_coord=y_range)

    new_fsp, new_xyz = [], []
    new_layer2neurons = {}

    if ratio:
      if not (0 <= x_range[0] < 1): raise ValueError("Min(x) should lie in (0, 1.0]!")
      if not (0 < x_range[1] <= 1): raise ValueError("Max(x) should lie in (0, 1.0]!")
      if not (0 <= y_range[0] < 1): raise ValueError("Min(y) should lie in (0, 1.0]!")
      if not (0 < y_range[1] <= 1): raise ValueError("Max(y) should lie in (0, 1.0]!")

      # In each layer, apply the bound to the neuron-related data
      for z, neurons_idx in dataset.layer2neurons.items():
        neuronXY_z = dataset.neuron_xyz[neurons_idx, :]
        minX, minY = np.min(neuronXY_z, axis=0)[:2]
        maxX, maxY = np.max(neuronXY_z, axis=0)[:2]
        lx = minX + x_range[0] * (maxX - minX)
        rx = minX + x_range[1] * (maxX - minX)
        ly = minY + y_range[0] * (maxY - minY)
        ry = minY + y_range[1] * (maxY - minY)
        # Segment the neurons within the rectangular region
        selected_neurons_idx = np.where((lx <= neuronXY_z[:, 0]) & (neuronXY_z[:, 0] <= rx) &
                                        (ly <= neuronXY_z[:, 1]) & (neuronXY_z[:, 1] <= ry))[0]  # wrt segmented neuronXY_z
        new_layer2neurons[z] = np.arange(len(new_xyz), len(new_xyz) + len(selected_neurons_idx), step=1)
        new_xyz.extend(neuronXY_z[selected_neurons_idx])
        new_fsp.extend(dataset.fsp[neurons_idx[selected_neurons_idx], :])

    else:
      for z, neurons_idx in dataset.layer2neurons.items():
        neuronXY_z = dataset.neuron_xyz[neurons_idx, :]
        minX, minY = np.min(neuronXY_z, axis=0)[:2]
        maxX, maxY = np.max(neuronXY_z, axis=0)[:2]
        lx = max(minX, x_range[0])
        rx = min(maxX, x_range[1])
        ly = max(minY, y_range[0])
        ry = min(maxY, y_range[1])
        # Segment the neurons within the rectangular region
        selected_neurons_idx = np.where((lx <= neuronXY_z[:, 0]) & (neuronXY_z[:, 0] <= rx) &
                                        (ly <= neuronXY_z[:, 1]) & (neuronXY_z[:, 1] <= ry))[0]
        new_layer2neurons[z] = np.arange(len(new_xyz), len(new_xyz) + len(selected_neurons_idx), step=1)
        new_xyz.extend(neuronXY_z[selected_neurons_idx])
        new_fsp.extend(dataset.fsp[neurons_idx[selected_neurons_idx], :])

    self.fsp = np.array(new_fsp)
    self.neuron_counts = self.fsp.shape[0]
    self.neuron_xyz = np.array(new_xyz)
    self.layer2neurons = new_layer2neurons

