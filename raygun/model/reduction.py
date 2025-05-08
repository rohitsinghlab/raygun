import torch
import torch.nn as nn
from einops import rearrange

class Reduce(nn.Module):
    """
    Reduce variable-length sequence embedding into fixed-length representation.

    Encodes a fixed number of segments of the original sequence as their average
    values. Optionally incorporate error-scaling term to add standard deviation
    scaled variation into embeddings for regularization and variational representation.
    """
    def __init__(self, num_segments=50, embed_dim=1280):
        self.num_segments = num_segments
        super(Reduce, self).__init__()

    def forward(self, embeddings, return_std=False):
        """
        Returns the reduced sequence representation and optionally the standard deviation
        used to scale variational noise.
        """
        batch_size, sequence_len, _ = embeddings.shape

        # calculate size of segments. normal size and gap-filling size
        segment_size = sequence_len // self.num_segments
        gap_segment_size = segment_size + 1

        # for sequences not perfectly divisible by num_segments, make segments
        # at beginning and end slighly larger to fill in the gap
        gap_segments = sequence_len - segment_size * self.num_segments
        segments_left = segments_right = gap_segments // 2

        # if gap is not even, add one more segment on left
        if gap_segments % 2 == 1:
            segments_left += 1
        middle_segments = self.num_segments - gap_segments

        # calculate lengths of 3 groups (left and right have +1 since they fill in the gap)
        len_left = segments_left * (gap_segment_size)
        len_middle = middle_segments * segment_size
        len_right = segments_right * (gap_segment_size)

        # find start and end indices of the 3 groups in the whole sequence
        left_start, left_end = 0, len_left
        middle_start, middle_end = len_left, len_left + len_middle
        right_start, right_end = len_left + len_middle, sequence_len

        # get embeddings for 3 groups
        left_embeddings = embeddings[:, left_start:left_end, :]
        middle_embeddings = embeddings[:, middle_start:middle_end, :]
        right_embeddings = embeddings[:, right_start:right_end, :]

        # get mean and standard deviation representations of each segment
        left_means, left_stds = self._calculate_mean_std(
            left_embeddings, segments_left, return_std
        )
        middle_means, middle_stds = self._calculate_mean_std(
            middle_embeddings, segments_left, return_std
        )
        right_means, right_stds = self._calculate_mean_std(
            right_embeddings, segments_left, return_std
        )
        embedding_means = torch.concat([left_means, middle_means, right_means], dim=1)
        embedding_stds = torch.concat([left_stds, middle_stds, right_stds], dim=1) if return_std else None

        # optionally return stds
        return (embedding_means, embedding_stds) if return_std else (embedding_means,)



    def _calculate_mean_std(self, embeddings, num_segments,
                            return_std=False):
        """
        Take a group of segments and extract the mean and optionally standard deviation
        of each segment as a compressed representation.
        """
        device = embeddings.device
        batch_size, _, embed_dim = embeddings.shape

        # if there are no segments in the group, then just return zeros
        if num_segments == 0:
            zero_tensor = torch.zeros(batch_size, 0, embed_dim).to(device)
            return (zero_tensor, zero_tensor) if return_std else (zero_tensor, None)

        # separate into segments
        embeddings_segmented = rearrange(embeddings, 'b (s l) e -> b s l e', s=num_segments)

        # calculate mean per segment
        segmented_means = torch.mean(embeddings_segmented, dim=2)

        # optionally return standard deviation
        # this can reprsent the amount of information lost in the reduction
        if return_std:
            segmented_stds = torch.std(embeddings_segmented, dim=2)
            return segmented_means, segmented_stds
        else:
            return segmented_means, None
