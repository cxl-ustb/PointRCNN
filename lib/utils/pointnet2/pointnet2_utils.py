import paddle
import paddle.nn as nn
import pointnet2


def furthest_point_sample(xyz, npoint):
    """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
    B, N, _ = xyz.shape
    paddle.device.set_device('gpu')
    output = paddle.zeros([B, npoint], dtype='int32')
    temp = paddle.zeros([B, N], dtype='float32')
    pointnet2.furthest_point_sampling_wrapper(xyz, temp, output, B, N, npoint)
    return output


def gather_operation(features, idx):
    """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
    B, npoint = idx.shape
    _, C, N = features.shape
    output = paddle.zeros([B, C, npoint], dtype='float32')

    output = pointnet2.gather_points_wrapper(features, idx, output, B, C, N, npoint)
    return output


def three_nn(unknown, known):
    """
        Find the three nearest neighbors of unknown in known
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
    B, N, _ = unknown.shape
    m = known.shape[1]
    dist2 = paddle.zeros([B, N, 3], dtype='float32')
    idx = paddle.zeros([B, N, 3], dtype='int32')

    dist2, idx = pointnet2.three_nn_wrapper(unknown, known, dist2, idx, B, N, m)

    return paddle.sqrt(dist2), idx


def three_interpolate(features, idx, weight):
    """
        Performs weight linear interpolation on 3 features
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """

    B, c, m = features.shape
    n = idx.shape[1]
    output = paddle.zeros([B, c, n], dtype='float32')

    output = pointnet2.three_interpolate_wrapper(features, idx, weight, output, B, c, m, n)
    return output


def grouping_operation(features, idx):
    """
                :param features: (B, C, N) tensor of features to group
                :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
                :return:
                    output: (B, C, npoint, nsample) tensor
                """

    B, nfeatures, nsample = idx.shape
    _, C, N = features.shape
    output = paddle.zeros([B, C, nfeatures, nsample])

    output = pointnet2.group_points_wrapper(features, idx, output, B, C, N, nfeatures, nsample)
    return output


def ball_query(radius, nsample, xyz, new_xyz):
    """
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
    B, N, _ = xyz.shape
    npoint = new_xyz.shape[1]

    idx = paddle.zeros([B, npoint, nsample], dtype='int32')
    idx = pointnet2.ball_query_wrapper(new_xyz, xyz, idx, B, N, npoint, radius, nsample)

    return idx


class QueryAndGroup(nn.Layer):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = paddle.transpose(xyz, [0, 2, 1])
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz -= paddle.unsqueeze(new_xyz.transpose([0, 2, 1]), -1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = paddle.concat([
                    grouped_xyz, grouped_features
                ], 1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Layer):
    def __init__(self, use_xyz):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = paddle.unsqueeze(paddle.transpose(xyz, [0, 2, 1]), axis=2)
        if features is not None:
            grouped_features = paddle.unsqueeze(features, 2)

            if self.use_xyz:
                new_features = paddle.concat([grouped_xyz, grouped_features], axis=1)
            else:
                new_features = grouped_features

        return new_features
