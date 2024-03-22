import numpy as np
import torch
from tonic.slicers import (
    slice_events_by_time,
)
import tonic.functional as tof
from typing import Any, List, Tuple
import random

class SliceByTimeEventsTargets:
    """
    Modified from tonic.slicers.SliceByTimeEventsTargets in the Tonic Library

    Slices an event array along fixed time window and overlap size. The number of bins depends
    on the length of the recording. Targets are copied.

    >        <overlap>
    >|    window1     |
    >        |   window2     |

    Parameters:
        time_window (int): time for window length (same unit as event timestamps)
        overlap (int): overlap (same unit as event timestamps)
        include_incomplete (bool): include the last incomplete slice that has shorter time
    """

    def __init__(self,time_window, overlap=0.0, seq_length=30, seq_stride=15, include_incomplete=False) -> None:
        self.time_window= time_window
        self.overlap= overlap
        self.seq_length=seq_length
        self.seq_stride=seq_stride
        self.include_incomplete=include_incomplete

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        t = data["t"]
        stride = self.time_window - self.overlap
        assert stride > 0

        if self.include_incomplete:
            n_slices = int(np.ceil(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        else:
            n_slices = int(np.floor(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        n_slices = max(n_slices, 1)  # for strides larger than recording time

        window_start_times = np.arange(n_slices) * stride + t[0]
        window_end_times = window_start_times + self.time_window
        indices_start = np.searchsorted(t, window_start_times)[:n_slices]
        indices_end = np.searchsorted(t, window_end_times)[:n_slices]

        if not self.include_incomplete:
            # get the strided indices for loading labels
            label_indices_start = np.arange(0, targets.shape[0]-self.seq_length, self.seq_stride)
            label_indices_end = label_indices_start + self.seq_length
        else:
            label_indices_start = np.arange(0, targets.shape[0], self.seq_stride)
            label_indices_end = label_indices_start + self.seq_length
            # the last label indices end should be the last label
            label_indices_end[-1] = targets.shape[0]

        assert targets.shape[0] >= label_indices_end[-1]

        return list(zip(zip(indices_start, indices_end), zip(label_indices_start, label_indices_end)))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ):
        return_data = []
        return_target = []
        for tuple1, tuple2 in metadata:
            return_data.append(data[tuple1[0]:tuple1[1]])
            return_target.append(targets[tuple2[0]:tuple2[1]])

        return return_data, return_target


class SliceLongEventsToShort:
    def __init__(self, time_window, overlap, include_incomplete):
        """
        Initialize the transformation.

        Args:
        - time_window (int): The length of each sub-sequence.
        """
        self.time_window = time_window
        self.overlap = overlap
        self.include_incomplete = include_incomplete

    def __call__(self, events):
        return slice_events_by_time(events, self.time_window, self.overlap, self.include_incomplete)


class Jitter:
    def __init__(self):
        """
        Initialize the transformation.

        Args:
        - time_window (int): The length of each sub-sequence.
        """
        pass

    def __call__(self, data, label):
        # x shift
        prob = 0.5
        p = [-1, +1]
        if random.random() > prob:
            x = max(1, int(random.random() * 10))
            x = x * p[random.random() > 0.5]
            data = np.concatenate([data[..., x:], data[..., :x]], axis=-1)
            label[..., 0] = (label[..., 0] - x / data.shape[-1]) % 1
        
        # y shift        
        if random.random() > prob:
            y = max(1, int(random.random() * 10))
            y = y * p[random.random() > 0.5]
            data = np.concatenate([data[..., y:, :], data[..., :y, :]], axis=-2)
            label[..., 1] = (label[..., 1] - y / data.shape[-2]) % 1

        # t shift
        if random.random() > prob:
            t = max(1, int(random.random() * 15))
            t = t * p[random.random() > 0.5]
            data = np.concatenate([data[t:], data[:t]], axis=0)
            label = np.concatenate([label[t:], label[:t]], axis=0)
        
        # x flip
        if random.random() > prob:
            data = np.flip(data, axis=-1)
            label[..., 0] = 1 - label[..., 0]
        
        # y flip
        if random.random() > prob:
            data = np.flip(data, axis=-2)
            label[..., 1] = 1 - label[..., 1]
        
        # # t flip
        # if random.random() > prob:
        #     data = np.flip(data, axis=0)
        #     label = label[::-1]

        # # spatial cutout
        # if random.random() > prob:
        #     h, w = (np.random.randint(6, high=12), np.random.randint(8, high=16))
        #     top = np.random.randint(4, data.shape[-2] - h + 1)
        #     left = np.random.randint(5, data.shape[-1] - w + 1)
        #     data[..., top:top+h, left:left+w] = 0

        return data.copy(), label.copy()

class EventSlicesToMap:
    def __init__(self, sensor_size, n_time_bins, per_channel_normalize, map_type='voxel'):
        """
        Initialize the transformation.

        Args:
        - sensor_size (tuple): The size of the sensor.
        - n_time_bins (int): The number of time bins.
        """
        self.sensor_size = sensor_size
        self.n_time_bins = n_time_bins
        self.per_channel_normalize = per_channel_normalize
        self.map_type = map_type

    def __call__(self, event_slices):
        """
        Apply the transformation to the given event slices.

        Args:
        - event_slices (Tensor): The input event slices.

        Returns:
        - Tensor: A batched tensor of voxel grids.
        """
        ev_maps = []
        for event_slice in event_slices:
            if self.map_type == 'voxel':
                ev_map = tof.to_voxel_grid_numpy(event_slice, self.sensor_size, self.n_time_bins)
            elif self.map_type == 'binary':
                ev_map = tof.to_frame_numpy(event_slice, self.sensor_size, n_time_bins=self.n_time_bins)
                ev_map = tof.to_bina_rep_numpy(ev_map, n_frames=1, n_bits=self.n_time_bins)
            elif self.map_type == 'frame':
                ev_map = tof.to_frame_numpy(event_slice, self.sensor_size, n_time_bins=self.n_time_bins)
            
            ev_map = ev_map.reshape(-1, ev_map.shape[-2], ev_map.shape[-1])

            # 归一化
            ev_map = (ev_map - ev_map.min()) / (ev_map.max() - ev_map.min())
            if self.per_channel_normalize:
                # Calculate mean and standard deviation only at non-zero values
                non_zero_entries = (ev_map != 0)
                for c in range(ev_map.shape[0]):
                    mean_c = ev_map[c][non_zero_entries[c]].mean()
                    std_c = ev_map[c][non_zero_entries[c]].std()

                    ev_map[c][non_zero_entries[c]] = (ev_map[c][non_zero_entries[c]] - mean_c) / (std_c + 1e-10)
            ev_maps.append(ev_map)
        return np.array(ev_maps).astype(np.float32)


class SplitSequence:
    def __init__(self, sub_seq_length, stride):
        """
        Initialize the transformation.

        Args:
        - sub_seq_length (int): The length of each sub-sequence.
        - stride (int): The stride between sub-sequences.
        """
        self.sub_seq_length = sub_seq_length
        self.stride = stride

    def __call__(self, sequence, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - sequence (Tensor): The input sequence of frames.
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of sub-sequences.
        - Tensor: A batched tensor of corresponding labels.
        """

        sub_sequences = []
        sub_labels = []

        for i in range(0, len(sequence) - self.sub_seq_length + 1, self.stride):
            sub_seq = sequence[i:i + self.sub_seq_length]
            sub_seq_labels = labels[i:i + self.sub_seq_length]
            sub_sequences.append(sub_seq)
            sub_labels.append(sub_seq_labels)

        return np.stack(sub_sequences), np.stack(sub_labels)
    

class SplitLabels:
    def __init__(self, sub_seq_length, stride):
        """
        Initialize the transformation.

        Args:
        - sub_seq_length (int): The length of each sub-sequence.
        - stride (int): The stride between sub-sequences.
        """
        self.sub_seq_length = sub_seq_length
        self.stride = stride
        # print(f"stride is {self.stride}")

    def __call__(self, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        sub_labels = []
        
        for i in range(0, len(labels) - self.sub_seq_length + 1, self.stride):
            sub_seq_labels = labels[i:i + self.sub_seq_length]
            sub_labels.append(sub_seq_labels)

        return np.stack(sub_labels)

class ScaleLabel:
    def __init__(self, scaling_factor):
        """
        Initialize the transformation.

        Args:
        - scaling_factor (float): How much the spatial scaling was done on input
        """
        self.scaling_factor = scaling_factor


    def __call__(self, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        labels[:,:2] =  labels[:,:2] * self.scaling_factor
        return labels
    
class TemporalSubsample:
    def __init__(self, temporal_subsample_factor):
        self.temp_subsample_factor = temporal_subsample_factor

    def __call__(self, labels):
        """
        temorally subsample the labels
        """
        interval = int(1/self.temp_subsample_factor)
        return labels[::interval]
    

class NormalizeLabel:
    def __init__(self, pseudo_width, pseudo_height):
        """
        Initialize the transformation.

        Args:
        - scaling_factor (float): How much the spatial scaling was done on input
        """
        self.pseudo_width = pseudo_width
        self.pseudo_height = pseudo_height
    
    def __call__(self, labels):
        """
        Apply normalization on label, with pseudo width and height

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        labels[:, 0] = labels[:, 0] / self.pseudo_width
        labels[:, 1] = labels[:, 1] / self.pseudo_height
        return labels

