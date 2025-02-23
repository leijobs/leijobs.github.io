---
layout: post
title: Motion Models
thumbnail-img: /assets/static/qofnk52b.png
tags: MOT
---

# MotionModels

## 运动模型

### 总结

| **模型**        | **适用场景**                                 | **优点**               | **缺点**                                                           |
| --------------- | -------------------------------------------- | ---------------------- | ------------------------------------------------------------------ |
| **CV**          | 目标速度恒定（如高速公路上的车辆）           | 简单，计算量小         | 无法处理加速度或转弯                                               |
| **CA**          | 目标加速度恒定（如城市道路上的车辆）         | 可以处理加速度         | 计算量较大，无法处理转弯                                           |
| **CTRV**        | 目标转弯且速度恒定（如车辆转弯、飞机盘旋）   | 可以处理转弯           | 无法处理加速度                                                     |
| **CTRA**        | 目标转弯且加速度变化（如车辆加速转弯）       | 可以处理转弯和加速度   | 计算复杂度较高                                                     |
| **IMM**         | 目标运动状态复杂（如车辆在城市道路中行驶）   | 适应性强，精度高       | 计算复杂度高                                                       |
| **Singer**      | 目标加速度随机（如行人或动物运动）           | 可以处理随机加速度     | 计算复杂度较高，需要调节参数                                       |
| **CS**          | 目标速度恒定但方向变化（如船只或无人机）     | 简单，计算量小         | 无法处理加速度                                                     |
| **NCA**         | 目标加速度近似恒定（如车辆在高速公路上行驶） | 可以处理小幅加速度变化 | 计算量较大                                                         |
| **Bicycle**     | 车辆运动预测（如自动驾驶车辆）               | 精度高，适用于车辆运动 | 假设车辆为刚性车身，忽略悬架和轮胎动力学不适用于高速或极端驾驶条件 |
| **Random Walk** | 目标运动无规律（如行人或动物）               | 简单，计算量小         | 预测精度低                                                         |

### 基本描述

#### CV（Constant Velocity，恒定速度模型）

- **描述**: 假设目标以恒定速度运动，速度和方向不变。
- **状态向量**: 位置和速度

$$
\mathbf{x}_{cv} = [x, y, v_x, v_y]
$$

- **状态方程**:

$$
\mathbf{\dot{x}}=
[v_x,v_y,0,0]
$$

- **离散化状态方程**:

$$
\mathbf{x_{k+1}}=
\begin{bmatrix}
1 & 0 & T & 0 \\
0 & 1 & 0 & T \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}\mathbf{x_{k}}
$$

#### CA（Constant Acceleration，恒定加速度模型）

- **描述**:  假设目标以恒定加速度运动，加速度不变。
- **状态向量**: 位置、速度和加速度

$$
\mathbf{x}_{ca} = [x, y, v_x, v_y, a_x, a_y]
$$

- **状态方程**:

$$
\mathbf{\dot{x}}=[v_x,v_y,a_x,a_y,0,0 ]
$$

- **离散化状态方程**:

$$
\mathbf{x_{k+1}}=
\begin{bmatrix}
1 & 0 & T & 0 & \frac{T^2}{2} & 0\\
0 & 1 & 0 & T & 0 & \frac{T^2}{2}\\
0 & 0 & 1 & 0 & T & 0 \\
0 & 0 & 0 & 1 & 0 & T \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}\mathbf{x_{k}}
$$

#### CTRV（Constant Turn Rate and Velocity，恒定转向率和速度模型）

- **描述**: 假设目标以恒定速度运动，速度和方向不变。
- **状态向量**: 位置、速度、航向角和转向率

$$
\mathbf{x}_{ctrv} = [x, y, v, \theta, \omega]
$$

- **状态方程**:

$$
\mathbf{\dot{x}}=
[v cos(\theta),v sin(\theta), 0, \omega, 0]
$$

- **离散化状态方程**:

$$
\mathbf{x_{k+1}}=
\begin{bmatrix}
x_k + \frac{v_k}{\omega_k}(sin(\theta_k + \omega_k T)-sin(\theta_k)) \\
y_k + \frac{v_k}{\omega_k}(-cos(\theta_k + \omega_k T) + cos(\theta_k)) \\
v_k \\
\theta_k + \omega_k T \\
\omega_k
\end{bmatrix}
$$

#### CTRA（Constant Turn Rate and Acceleration，恒定转向率和加速度模型）

- **描述**: 假设目标以恒定转向率和恒定加速度运动。
- **状态向量**: 位置、速度、航向角、转向率和加速度

$$
X_{ctra} = [x, y, v, \theta, \omega, a]
$$

- **状态方程**:

$$
\mathbf{\dot{x}}=
[v cos(\theta),v sin(\theta), a, \omega, 0, 0]
$$

- **离散化状态方程**:

$$
\mathbf{x_{k+1}}=
\begin{bmatrix}
x_k + \frac{v_k}{\omega_k}(sin(\theta_k + \omega_k T)-sin(\theta_k)) + \frac{a_k T^2}{2}cos(\theta_k + \omega_k T) \\
y_k + \frac{v_k}{\omega_k}(-cos(\theta_k + \omega_k T) + cos(\theta_k)) + \frac{a_k T^2}{2}sin(\theta_k + \omega_k T)\\
v_k + a_k T\\
\theta_k + \omega_k T \\
\omega_k \\
a_k
\end{bmatrix}
$$

#### IMM**（Interacting Multiple Model，交互多模型）**

- **描述**: 结合多个运动模型（如 CV、CA、CTRV），根据目标运动状态动态切换模型。
- **状态向量**:

$$
X_{imm} = a_1 X_{cv} + a_2 X_{ca} + a_3 X_{ctrv} + a_4 X_{ctra} + \dots
$$

#### Singer

- **描述**: 假设目标的加速度是一个随机过程（通常建模为一阶马尔可夫过程）。
- **基本假设**：

  - 目标的加速度$a(t)$是一个随机过程，变化服从一阶马尔可夫过程，即：

  $$
  ot{a}(t) = -\alpha a(t) + \omega(t)
  $$
- **状态向量**: 位置、速度、加速度

$$
\mathbf{x}_{singer}=
[x,\dot{x},\ddot{x}]
$$

- **状态方程**:

$$
\mathbf{\dot{x}}=\mathbf{F}\mathbf{x}+\mathbf{G}\mathbf{w}=
\begin{bmatrix}0&1&0\\0&0&1\\0&0&-\alpha \end{bmatrix}\mathbf{x_{k}} + \begin{bmatrix}0\\0\\1 \end{bmatrix} \mathbf{w}(t)
$$

其中，

> - $\mathbf{F}$是状态转移矩阵。
> - $\mathbf{G}$是噪声输入矩阵。

- **离散化状态方程：**

$$
\mathbf{x}_{k+1} = \mathbf{F}_d \mathbf{x}_k+\mathbf{w}_k
$$

其中：

> - $\mathbf{F}_d$是状态转移方程的离散化形式:
>
> $$
> \mathbf{F}_d=\left[\begin{array}{lll}
1 & T & \frac{T^2}{2} \\
0 & 1 & T \\
0 & 0 & e^{-\alpha T}
\end{array}\right]$$

> 其中，$T$是时间步长。
>
> - $\mathbf{w}_k$是离散时间过程噪声，其协方差矩阵为：
>
> $$
> \mathbf{Q}_{k}=2 \alpha \sigma_{a}^{2}\left[\begin{array}{lll}
q_{11} & q_{12} & q_{13} \\
q_{21} & q_{22} & q_{23} \\
q_{31} & q_{32} & q_{33}
\end{array}\right]$$

> $\sigma_a^2$是加速度的方差
> 其中：
>
> $$
> \begin{array}{rlr}
q_{11}=\frac{T^{b}}{20}, & q_{12}=\frac{T^{4}}{8}, & q_{13}=\frac{T^{3}}{6} \\
q_{21}=\frac{T^{4}}{8}, & q_{22}=\frac{T^{3}}{3}, & q_{23}=\frac{T^{2}}{2} \\
q_{31}=\frac{T^{3}}{6}, & q_{32}=\frac{T^{2}}{2}, & q_{33}=T
\end{array}$$

- **适用场景**: 目标加速度变化随机（如行人或动物运动），需要处理随机加速度的短期预测。

#### CS（Constant Speed，恒定速度模型）

- **描述**: 假设目标以恒定速度运动，但方向可以变化。
- **状态向量**: 位置、速度和航向角
- **适用场景**: 目标加速度变化随机（如行人或动物运动），需要处理随机加速度的短期预测。

$$
X_{cs} = [x, y, v, \theta]
$$

- **状态方程**:

$$
\mathbf{\dot{x}}=
[v cos(\theta),v sin(\theta), 0, \omega]
$$

- **离散化状态方程**:

$$
\mathbf{x_{k+1}}=
\begin{bmatrix}
x_k + \frac{v_k}{\omega_k}(sin(\theta_k + \omega_k T)-sin(\theta_k)) \\
y_k + \frac{v_k}{\omega_k}(-cos(\theta_k + \omega_k T) + cos(\theta_k)) \\
0 \\
\omega_k
\end{bmatrix}
$$

#### NCA（Nearly Constant Acceleration，近似恒定加速度模型）

- **描述**: 假设目标的加速度近似恒定，但允许小幅变化。NCA 模型是 **Constant Acceleration (CA) 模型** 的扩展，通过引入过程噪声来模拟加速度的随机变化。
- **状态向量**: 位置、速度和加速度

$$
X_{nca} = [x, y, v_x, v_y, a_x, a_y]
$$

- **状态方程**:

$$
\mathbf{\dot{x}}=[v_x,v_y,a_x,a_y,0,0 ] + \mathbf{w}(t)
$$

- **离散化状态方程**:

$$
\mathbf{x}_{k+1} = \mathbf{F}_d \mathbf{x}_k+\mathbf{w}_k
$$

其中：

> - $\mathbf{F}_d$是状态转移方程的离散化形式:
>
> $$
> \mathbf{F}_d=\begin{bmatrix}
1 & 0 & T & 0 & \frac{T^2}{2} & 0\\
0 & 1 & 0 & T & 0 & \frac{T^2}{2}\\
0 & 0 & 1 & 0 & T & 0 \\
0 & 0 & 0 & 1 & 0 & T \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

> 其中，$T$是时间步长。
>
> - $\mathbf{w}_k$是离散时间过程噪声，其协方差矩阵为：
>
> $$
> \mathbf{Q}_{k}= \sigma_{a}^{2}\left[\begin{array}{lll}
\frac{T^5}{20} & 0 & \frac{T^4}{8} & 0 & \frac{T^3}{6} & 0\\
0 & \frac{T^5}{20} & 0 & \frac{T^4}{8} & 0 & \frac{T^3}{6}\\
\frac{T^4}{8} & 0 & \frac{T^3}{3} & 0 & \frac{T^2}{2} & 0 \\
0 & \frac{T^4}{8} & 0 & \frac{T^3}{3} & 0 & \frac{T^2}{2} \\
\frac{T^3}{6} & 0 & \frac{T^2}{2} & 0 & T & 0 \\
0 & \frac{T^3}{6} & 0 & \frac{T^2}{2} & 0 & T
\end{array}\right]$$

> 其中，$\sigma_a^2$是加速度的方差

#### Bicycle

- **描述**: 基于车辆的动力学模型，考虑前轮转向角和车辆几何。
- **基本假设**:

  - 车辆简化为一个两轮自行车，前轮负责转向，后轮负责驱动。
  - 车辆的转向角$\sigma$是前轮相对于车辆纵轴的转角。
  - 车辆的几何关系由轴距$L$（前轮和后轮之间的距离）决定。
- **状态向量**: 位置、速度和加速度

$$
\mathbf{x}_{bicycle} = [x, y, \theta, v]
$$

- **状态方程：**

$$
\dot{\mathbf{x}}(t)=
\begin{bmatrix}
\dot{x}(t) \\
\dot{y}(t) \\
\dot{\theta}(t) \\
\dot{v}(t)
\end{bmatrix}=
\begin{bmatrix}
v(t)\cos(\theta(t)) \\
v(t)\sin(\theta(t)) \\
\frac{v(t)}{L}\tan(\delta(t)) \\
a(t)
\end{bmatrix}
$$

其中，

> - $\dot{x}(t)$和$\dot{y}(t)$是车辆在$x$和$y$方向的速度。
> - $\dot{\theta}(t)$是车辆的角速度。
> - $\dot{v}(t) = a(t)$是车辆的加速度。

- **离散化状态方程：**

$$
\mathbf{x}_{k+1} = \mathbf{x}_k+\mathbf{f}(\mathbf{x}_k,u_k) T
$$

其中：

> - $\mathbf{f}(\mathbf{x}_k, \mathbf{u}_k)$是连续时间状态方程的离散化形式:
>
> $$
> \mathbf{f}(\mathbf{x}_k,\mathbf{u}_k)=
\begin{bmatrix}
v_k\cos(\theta_k) \\
v_k\sin(\theta_k) \\
\frac{v_k}{L}\tan(\delta_k) \\
a_k
\end{bmatrix}$$

> - $\mathbf{u}_k = [\delta_k,a_k]^T$是控制输入，包括转向角$\delta_k$和加速度$\alpha_k$。

#### Random Walk**（随机游走**）

- **描述**:  假设目标的位置随机变化，没有固定的运动规律，需要处理随机运动的短期预测。
- **状态向量**:

$$
\mathbf{x}_{rw} = [x, y]
$$

- **状态方程**:

$$
\mathbf{x} = [0, 0] + \mathbf{w}(t)
$$
