import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.unpool import knn_interpolate

# From: https://github.com/Strawberry-Eat-Mango/PCT_Pytorch


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = (
        torch.eye(num_tokens)
        .expand(batch_size, num_tokens, num_tokens)
        .to(all_layer_matrices[0].device)
    )
    all_layer_matrices = [
        all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))
    ]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer + 1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


def get_dists(points1, points2):
    """
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    """
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + torch.sum(
        torch.pow(points2, 2), dim=-1
    ).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(
        dists < 0, torch.ones_like(dists) * 1e-7, dists
    )  # Very Important for dist = 0.
    return torch.sqrt(dists).float()


def fps(xyz, M):
    """
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B,), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :]  # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    xyz = xyz.contiguous()

    fps_idx = fps(xyz, npoint).long()  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    # idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat(
        [grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)],
        dim=-1,
    )
    return new_xyz, new_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm1d(out_channels)
        self.bn2 = BatchNorm1d(out_channels)
        self.act1 = ReLU()
        self.act2 = ReLU()
        self.pool = AdaptiveMaxPool1d(1)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        self.x_size_1 = x.size()
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.act1(self.bn1(self.conv1(x)))  # B, D, N
        x = self.act2(self.bn2(self.conv2(x)))  # B, D, N
        x = self.pool(x)
        self.x_size_2 = x.size()
        x = x.view(batch_size, -1).reshape(b, n, -1).permute(0, 2, 1)
        return x

    def relprop(self, cam, **kwargs):
        cam = cam.permute(0, 2, 1)
        cam = cam.reshape(self.x_size_2)
        cam = self.pool.relprop(cam, **kwargs)

        cam = self.act2.relprop(cam, **kwargs)
        cam = self.bn2.relprop(cam, **kwargs)
        cam = self.conv2.relprop(cam, **kwargs)
        cam = self.act1.relprop(cam, **kwargs)
        cam = self.bn1.relprop(cam, **kwargs)
        cam = self.conv1.relprop(cam, **kwargs)

        cam = cam.reshape(self.x_size_1)
        cam = cam.permute(0, 1, 3, 2)

        return cam


class PCT(nn.Module):
    def __init__(self, output_channels=40):
        super(PCT, self).__init__()
        self.conv1 = Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = BatchNorm1d(64)
        self.bn2 = BatchNorm1d(64)
        self.act1 = ReLU()
        self.act2 = ReLU()
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last()

        self.cat = Cat()

        self.conv_fuse = Sequential(
            Conv1d(1280, 1024, kernel_size=1, bias=False),
            BatchNorm1d(1024),
            LeakyReLU(negative_slope=0.2),
        )

        self.pool = AdaptiveMaxPool1d(1)

        self.linear1 = Linear(1024, 512, bias=False)
        self.bn6 = BatchNorm1d(512)
        self.act3 = LeakyReLU(negative_slope=0.2)
        self.dp1 = Dropout(p=0.5)
        self.linear2 = Linear(512, 256)
        self.bn7 = BatchNorm1d(256)
        self.act4 = LeakyReLU(negative_slope=0.2)
        self.dp2 = Dropout(p=0.5)
        self.linear3 = Linear(256, output_channels)

    def forward(self, x):  # b,3,1024
        self.orig_xyz = x
        xyz = x.permute(0, 2, 1)  # b,1024,3
        self.batch_size, _, _ = x.size()
        # B, D, N
        x = self.act1(self.bn1(self.conv1(x)))  # b,64,1024
        # B, D, N
        x = self.act2(self.bn2(self.conv2(x)))  # b,64,1024
        x = x.permute(0, 2, 1)  # b,64,1024

        self.new_xyz_1, new_feature = sample_and_group(
            npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x
        )  # b,512,32,128

        feature_0 = self.gather_local_0(new_feature)  # b,128,512

        feature = feature_0.permute(0, 2, 1)  # b,512,128

        self.new_xyz_2, new_feature = sample_and_group(
            npoint=256, radius=0.2, nsample=32, xyz=self.new_xyz_1, points=feature
        )  # b,256,32,256
        feature_1 = self.gather_local_1(new_feature)  # 1, 256, 256

        x = self.pt_last(feature_1)
        x = self.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        self.fuse_shape = x.shape
        x = self.pool(x).view(self.batch_size, -1)
        x = self.act3(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.act4(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x

    def relprop(
        self,
        cam=None,
        method="transformer_attribution",
        is_ablation=False,
        start_layer=0,
        **kwargs,
    ):
        cam = self.linear3.relprop(cam, **kwargs)
        cam = self.dp2.relprop(cam, **kwargs)
        cam = self.act4.relprop(cam, **kwargs)
        cam = self.bn7.relprop(cam, **kwargs)
        cam = self.linear2.relprop(cam, **kwargs)
        cam = self.dp1.relprop(cam, **kwargs)
        cam = self.act3.relprop(cam, **kwargs)
        cam = self.bn6.relprop(cam, **kwargs)
        cam = self.linear1.relprop(cam, **kwargs)

        cam = torch.unsqueeze(cam, -1)

        cam = self.pool.relprop(cam, **kwargs)

        cam = self.conv_fuse.relprop(cam, **kwargs)
        cam, cam_1 = self.cat.relprop(cam, **kwargs)
        cam = self.pt_last.relprop(cam, **kwargs)

        if method == "full":
            cam = self.gather_local_1.relprop(cam, **kwargs)
            cam = cam.sum(2).reshape(self.batch_size, 128, 512)
            cam = self.gather_local_0.relprop(cam, **kwargs)
            cam = cam.sum(2).reshape(self.batch_size, 64, 1024)

            cam = self.act2.relprop(cam, **kwargs)
            cam = self.bn2.relprop(cam, **kwargs)
            cam = self.conv2.relprop(cam, **kwargs)

            cam = self.act1.relprop(cam, **kwargs)
            cam = self.bn1.relprop(cam, **kwargs)
            cam = self.conv1.relprop(cam, **kwargs)
            return cam.squeeze(0)

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for attn in [
                self.pt_last.sa1,
                self.pt_last.sa2,
                self.pt_last.sa3,
                self.pt_last.sa4,
            ]:
                attn_heads = attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)

            rollout = compute_rollout_attention(attn_cams, start_layer=start_layer)
            rollout = knn_interpolate(
                rollout.squeeze(0),
                self.new_xyz_2.squeeze(0),
                self.orig_xyz.permute(0, 2, 1).squeeze(0),
            )
            rollout = rollout.sum(1)
            return rollout

        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for attn in [
                self.pt_last.sa1,
                self.pt_last.sa2,
                self.pt_last.sa3,
                self.pt_last.sa4,
            ]:
                grad = attn.get_attn_gradients()
                cam = attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            rollout = knn_interpolate(
                rollout.squeeze(0),
                self.new_xyz_2.squeeze(0),
                self.orig_xyz.permute(0, 2, 1).squeeze(0),
            )
            rollout = rollout.sum(1)
            return rollout

        elif method == "last_layer_attn":
            cam = self.pt_last.sa4.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = knn_interpolate(
                cam,
                self.new_xyz_2.squeeze(0),
                self.orig_xyz.permute(0, 2, 1).squeeze(0),
            )
            cam = cam.sum(1)
            return cam


class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = BatchNorm1d(channels)
        self.bn2 = BatchNorm1d(channels)

        self.act1 = ReLU()
        self.act2 = ReLU()

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)
        self.cat = Cat()

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = self.cat((x1, x2, x3, x4), dim=1)

        return x

    def relprop(self, cam, **kwargs):
        cam1, cam2, cam3, cam4 = self.cat.relprop(cam, **kwargs)

        cam3 = self.sa4.relprop(cam4, **kwargs)
        cam2 = self.sa3.relprop(cam3, **kwargs)
        cam1 = self.sa2.relprop(cam2, **kwargs)
        cam = self.sa1.relprop(cam1, **kwargs)

        cam = self.conv2.relprop(cam, **kwargs)
        cam = self.bn2.relprop(cam, **kwargs)
        cam = self.act2.relprop(cam, **kwargs)
        cam = self.conv1.relprop(cam1, **kwargs)
        cam = self.bn1.relprop(cam, **kwargs)
        cam = self.act1.relprop(cam, **kwargs)

        return cam


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.clone = Clone()
        self.q_conv = Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.matmul1 = bmm()
        self.matmul2 = bmm()

        self.v_conv = Conv1d(channels, channels, 1)
        self.trans_conv = Conv1d(channels, channels, 1)
        self.after_norm = BatchNorm1d(channels)
        self.act1 = ReLU()
        self.softmax = Softmax(dim=-1)
        self.add1 = Add()
        self.add2 = Add()

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.clone(x, 5)
        # b, n, c
        x_q = self.q_conv(x1).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x2)
        x_v = self.v_conv(x3)
        self.save_v(x_v)
        # b, n, n
        energy = self.matmul1([x_q, x_k])

        attn = self.softmax(energy)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))
        # b, c, n

        self.save_attn(attn)
        if attn.requires_grad:
            attn.register_hook(self.save_attn_gradients)

        x_r = self.matmul2([attn, x_v])

        x_r = self.add1([x4, -x_r])
        x_r = self.trans_conv(x_r)
        x_r = self.after_norm(x_r)
        x_r = self.act1(x_r)
        x = self.add2([x5, x_r])
        return x

    def relprop(self, cam, **kwargs):
        cam_5, cam_r = self.add2.relprop(cam, **kwargs)
        cam_r = self.act1.relprop(cam_r, **kwargs)
        cam_r = self.after_norm.relprop(cam_r, **kwargs)
        cam_r = self.trans_conv.relprop(cam_r, **kwargs)
        cam_4, cam_r = self.add1.relprop(cam_r, **kwargs)

        cam1, cam_v = self.matmul2.relprop(-cam_r, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        cam1 = self.softmax.relprop(cam1, **kwargs)
        cam_q, cam_k = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        cam_3 = self.v_conv.relprop(cam_v, **kwargs)
        cam_2 = self.k_conv.relprop(cam_k, **kwargs)

        cam_q = cam_q.permute(0, 2, 1)
        cam_1 = self.q_conv.relprop(cam_q, **kwargs)
        cam = self.clone.relprop((cam_1, cam_2, cam_3, cam_4, cam_5), **kwargs)
        return cam


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha):
        return R


class RelPropSimple(RelProp):
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs


class ReLU(nn.ReLU, RelProp):
    pass


class GELU(nn.GELU, RelProp):
    pass


class LeakyReLU(nn.LeakyReLU, RelProp):
    pass


class Softmax(nn.Softmax, RelProp):
    pass


class Dropout(nn.Dropout, RelProp):
    pass


class MaxPool1d(nn.MaxPool1d, RelPropSimple):
    pass


class LayerNorm(nn.LayerNorm, RelProp):
    pass


class AdaptiveMaxPool1d(nn.AdaptiveMaxPool1d, RelPropSimple):
    pass


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs


class bmm(RelPropSimple):
    def __init__(self):
        super().__init__()

    def forward(self, operands):
        return torch.bmm(operands[0], operands[1])


class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__("num", num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R


class Cat(RelProp):
    def forward(self, inputs, dim):
        self.__setattr__("dim", dim)
        return torch.cat(inputs, dim)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs


class Sequential(nn.Sequential):
    def relprop(self, R, alpha):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R


class BatchNorm1d(nn.BatchNorm1d, RelProp):
    def relprop(self, R, alpha):
        X = self.X
        beta = 1 - alpha
        if len(R.shape) == 3:
            weight = self.weight.unsqueeze(0).unsqueeze(2) / (
                (self.running_var.unsqueeze(0).unsqueeze(2).pow(2) + self.eps).pow(0.5)
            )
        else:
            weight = self.weight.unsqueeze(0) / (
                (self.running_var.unsqueeze(0).pow(2) + self.eps).pow(0.5)
            )
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R


class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R


class Conv1d(nn.Conv1d, RelProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
            (Z.size()[2] - 1) * self.stride[0]
            - 2 * self.padding[0]
            + self.kernel_size[0]
        )

        return F.conv_transpose1d(
            DY,
            weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
        )

    def relprop(self, R, alpha):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = (
                self.X * 0
                + torch.min(
                    torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True
                )[0]
            )
            H = (
                self.X * 0
                + torch.max(
                    torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True
                )[0]
            )
            Za = (
                torch.conv1d(
                    X, self.weight, bias=None, stride=self.stride, padding=self.padding
                )
                - torch.conv1d(
                    L, pw, bias=None, stride=self.stride, padding=self.padding
                )
                - torch.conv1d(
                    H, nw, bias=None, stride=self.stride, padding=self.padding
                )
                + 1e-9
            )

            S = R / Za
            C = (
                X * self.gradprop2(S, self.weight)
                - L * self.gradprop2(S, pw)
                - H * self.gradprop2(S, nw)
            )
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv1d(
                    x1, w1, bias=None, stride=self.stride, padding=self.padding
                )
                Z2 = F.conv1d(
                    x2, w2, bias=None, stride=self.stride, padding=self.padding
                )
                S1 = safe_divide(R, Z1)
                S2 = safe_divide(R, Z2)
                C1 = x1 * self.gradprop(Z1, x1, S1)[0]
                C2 = x2 * self.gradprop(Z2, x2, S2)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R
