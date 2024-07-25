---
layout: post
title: Mono-Semantic-Maps
thumbnail-img: ./assets/static/O2mSbWrb4osQzcxcP9ocs8RSnjQ.png
tags: BEV Perception Mono-view
---

# Mono-Semantic-Map

## Info

> 论文：[https://arxiv.org/abs/2003.13402](https://arxiv.org/abs/2003.13402)
> 
> github：[https://github.com/tom-roddick/mono-semantic-maps](https://github.com/tom-roddick/mono-semantic-maps)

## Framework

### Abstract

自动驾驶通常依赖于精细的 BEV 地图来获得动态和静态要素。实时生成地图表示是融合很多视觉要素的复杂过程，包括地面估计，道路分割和 3D 检测。

作者提出了基于单目的简单，统一的地图预测方法。地图本身使用语义贝叶斯占据网格网络来累积多帧时序特征，算法在 Nuscenes 和 ArgoVerse 进行测试，state-of-the-art 的性能表现。

### Intro

自动驾驶和机器人平台通常依赖于信息丰富且细节的环境表示，包括动态和静态要素。以上环境表示是后续规划控制的基础。

当前的主流方法是 BEV 视图，能够简洁有效的描述空间信息，并有利于后续处理，因为导航信息往往依赖于地平面。

BEV 地图的构建是一个多阶段的复杂过程，依包括 SFM，地平面估计，道路分割等等。直觉上，这些所有 task 都是相关的：可根据地面信息查找车辆，车辆出现则代表对应区域存在道路，因此端到端方法似乎是可行的。

本文章中作者致力于基于单张图像生成语义 BEV 地图，不依赖于 Lidar 和 Radar，这种建图能力对于自动驾驶车辆至关重要。作者选择概率占据网格地图来表示，易于表示多模态和 temporal 信息，并适合 CNN 处理和学习。作者将传统网格扩展为语义网格信息，描述了目标网格的类别，然后目标是预测每个 BEV 位置的语义类别的置信度。

贡献如下：

1. Novel dense transformer 层来将 image 特征图 map 到 BEV 地图
2. 多尺度金字塔卷积网络来提取特征，用于预测精准的 BEV 视图
3. 在 nuscenes 和 argoverse 进行验证

最后量化分析了贝叶斯语义占据网格能用于累计多帧多相机的信息从而构建完整的地图，算法能够实时推理，在 2080TI 上的推理速度可达 23.3 帧。

### Related Work

#### Map representations for autonomous driving

高精度 BEV 地图表示对于各种任务都至关重要。HDNet 使用地面高度先验来提升 Lidar 点云的质量。【18】将稀疏 HD 地图和视觉观测相结合用于实现高精度定位。

BEV 地图天生适合预测和规划：【4，9】将本地环境渲染为 BEV 表示，包括道路几何，车道线以及交通参与者，并预测他们的轨迹。类似表示在【2】中作为模仿学习的输入，从而预测目标的未来状态。【12】使用地图来增强端到端驾驶模型，提升了驾驶表现。

#### Top-down representations from images

大量方法来处理单目到 BEV 的投影过程。一般方法是 IPM，通过单应矩阵将 PV 转化为 BEV 视图。其他工作致力于在 BEV 空间进行 3D 检测任务，将 2D 检测 mapping 到 3D 视图，或者在 BEV 空间直接预测 3D bbox。

较少工作处理了语义 BEV 的问题。有些算法使用 IPM 再进行分割，或者 RGBD 预测地图。VED 使用 encoder-decoder 来预测 BEV 视图，这种全卷积导致空间信息丢失，因此输出结果相对粗糙并且难以预测小目标。CrossView 策略类似，使用全连接 view-transformer 实现，【24】使用 in-painting CNN 来推理语义 label 和前景目标的深度信息，基于语义点云深度 BEV 图像。

由于缺乏 GT 数据，大部分方法依赖于双目或者对齐不精准的地图。基于真实视觉来验证性能至关重要。

### Semantic occupancy grid prediction

占据网格是一种随机离散场，其中每个位置$x_i$都关联了一个状态$m_i$，比如占据（$m_i = 1$）或者 free（$m_i = 0$）。实际上，世界的真实状态是未知的，因此$m_i$表示为一组随机数，描述网格的预测状态（$p(m_i \vert z_{1:t})$），其中$z_{1:t}$表示一组观测。

这个网格信息可以进一步扩展为语义网格，此时的预测状态不再是一个值，作者使用 deep CNN 来做 inverse sensor model，表示为$p(m_i^c \vert z_{t}) = f_\theta(z_t,x_i)$，设置为对每个类别的预测状态。

因此，目标即为预测 2D BEV 图像的多类别的 binary label。看起来和语义分割类似，但是实际上 PV 和 BEV 视图的坐标系表示不一致，因此作者引入 transformer layer，利用视觉集合和全连接推理转化为为 BEV 特征。

该 dense transformer layer 作为 PyrOccNet 的一部分，该包含四个阶段：

1. backbone：生成多尺度语义和几何特征
2. FPN-based 金字塔：上采样生成更高分辨率的上下文信息
3. Dense Transformer：将图像特征汇集为 BEV 特征
4. Topdown Network：预测语义网格的概率

![](/assets/static/O2mSbWrb4osQzcxcP9ocs8RSnjQ.png)

#### Losses

Losses 分为两部分，确定目标的 binary 交叉熵损失和不确定目标的不确定性损失：

- binary 交叉熵损失

binary 交叉熵损失有利于语义网格的预测概率$p(m_i\vert z_{1:t})$到 gt 网格$\hat{m}^c_i$的回归。由于数据集中包含大量小目标，因此 loss 为 binary 交叉熵 loss 的平衡变种，其中$\alpha^c$表示平衡类别$\alpha$的尺度系数。

$$
\mathcal{L}_{xent}=\alpha^c\hat{m}_i^c\log p(m_i^c\vert z_t)+(1-\alpha^c)(1-\hat{m}_i^c)\log\left(1-p(m_i^c\vert z_t)\right)
$$

- 不确定目标的不确定性 loss

对于不确定的网格，网络仍然会预测较高的置信度。为了引导网络对不确定的网格预测不确定性，这种引入最大化预测熵的 loss，引导网络对于每个类别的预测靠近 0.5：

$$
\mathcal{L}_{uncert}=1-p(m_i^c\vert z_t)\log_2p(m_i^c \vert z_t)
$$

对于网络不可见的网格，使用最大熵损失。此时的网格要么在 FOV 以外，要么被遮挡了，对于该区域的网格忽略交叉熵损失。

最终 loss 等于两者相加，如下：

$$
\mathcal{L}_{total} = \mathcal{L}_{xent} + \lambda \mathcal{L}_{uncert}
$$

其中，$\lambda = 0.001$表示常量权重因子

#### Temporal and sensor data fusion

贝叶斯占据网格通过贝叶斯滤波器提供了对多帧多组观测的天然的融合方法。假设得到图像观测$z_t$时对应的外参为$M_t$，可以将网格概率表示$p(m^c_i\vert z_t)$为 logistic 函数的反函数（log-odds）表示：

$$
l^c_{i,t} = \frac{p(m^c_i\vert z_t)}{1 - p(m^c_i\vert z_t)}
$$

方便地等同于网络的前象限输出激活。因此，观测值 1 至 t 的合并对数占空比为:

$$
l^c_{i,1:t} = l^c_{i,1:t-1} + l^c_{i,t} - l^c_0
$$

应用标准 sigmoid 函数，可以恢复出融合后的占用概率：

$$
p(m^c_i\vert z_{1:t}) = \frac{1}{1+exp(-l^c_{i,1:t})}
$$

log-odds 值$l^c_0$表示网格中类别$c$的先验概率：

$$
l^c_0 = \frac{p(m^c_i)}{1-p(m^c_i)}
$$

为了获得全局坐标系下的占据概率，使用外参矩阵将在本地相机帧坐标系中预测占用率的网络输出重新采样到全局帧中，比如$p(m^c_i\vert z_t) = f_{\theta}(z_t, M^{-1}_tx_i)$。该方法能够聚合环视信息以及融合超过 20s 的网格观测结果。

#### Dense transformer layer

占格预测任务面临的基本挑战之一是输入和输出存在于两个完全不同的坐标系中：透视图像空间和正交 BEV 空间。为了解决这个问题，我们引入了一个简单的转换层，如图所示。我们的目标是将维度为$C \times H \times W$的图像平面特征图转换为$C \times Z \times X$的 BEV 视图特征。

![](./assets/static/P65GboghZoS92ixdmPrc5r6Xnth.png)

dense transformer 层的设计灵感来源于以下观察：虽然网络需要大量的垂直背景信息才能将特征映射到鸟瞰图上（由于遮挡、缺乏深度信息以及未知的地面拓扑结构），但在水平方向上，BEV 位置与图像位置之间的关系可以通过简单的相机几何图形建立起来。

因此，为了最大限度地保留空间信息，我们将图像特征图的垂直维度和通道维度折叠成一个大小为 B 的 bottleneck，但保留水平维度 W。然后，我们沿水平轴进行一维卷积，重塑得到的特征图，得到一个尺寸为$C \times Z \times W$ 的张量。然而，由于透视的原因，这个仍采用图像空间坐标的特征图实际上对应于 BEV 空间中的一个梯形，因此最后一步是使用已知的相机焦距 $f_x$ 和水平偏移 $u_0$ 将其重新采样到笛卡尔框架中。

> 分析：该方法类似于 OFT，相当于假设各个维度深度一致

#### Multiscale transformer pyramid

![](./assets/static/J4Ivb59uwoHmikxTPHAcK7D2nJf.png)

上节所述的重采样步骤包括，在距离摄像机$z$ 远的一行网格单元中，以下列间隔对特征图进行采样:

$$
\Delta u=\frac{f\Delta x}{sz}
$$

其中，$\Delta x$表示网格分辨率，$s$为输入特征相对于图像的下采样系数。常量系数的设置是个问题：距离摄像机较远的网格单元对应的特征会变得模糊，而距离摄像机较近的网格单元对应的特征则会采样不足，从而出现混叠现象。因此，建议使用多个 transformer，作用于具有下采样因子 $s_k = 2^{k+3},k \in \{0,...,4\}$的特征图金字塔。第 $k$个 transformer 生成深度值子集的特征，范围从$z_k$ 到 $z_{k-1}$，其中 $z_k$ 由以下公式给出：

$$
z_k = \frac{f\Delta x}{s_k}
$$

表列出了典型相机和网格设置的值。

![](./assets/static/MXKlbLfghomiZ6x2TV6cDKvjnIb.png)

然后，将每个 transformer 沿深度轴的输出连接起来，就能绘制出最终的鸟瞰特征图。这种方法的一个缺点是，在高分辨率下，特征图的高度 Hk 可能会变得非常大，从而导致相应的 dense transformer 中参数数量过多。但在实际操作中，我们可以将特征图的高度裁剪：

$$
H_k = f \frac{y_{max} - y_{min}}{s_k z_k}
$$

对应于世界空间中$y_{max}$和 $y_{min}$之间的固定垂直范围。这意味着裁剪后的特征图在不同尺度下高度基本保持不变。特征图取自骨干网络中从 conv3 到 conv7 的每个残差阶段的输出。为了确保高分辨率的特征图仍能涵盖较大的空间范围，我们采用以下方法从较低分辨率的特征图中添加了上采样层。

> 分析：该种 view transformer + crop + concat 的方法虽然可以拼接出完整的 BEV 视图，但是将导致不同区域的特征分辨率不同，不是线性的而是阶跃的变化
