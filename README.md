# 勇气

> 小游戏《勇气》的代码实现。
>
> 基于 CFR 的纳什均衡求解器，含自适应层。
>
> 算法仍在快速迭代中，未来的整体架构可能会随时发生变化。

## 游戏形式化

### 状态空间

游戏是 **完全信息同时行动零和随机博弈**。每一时刻的状态定义为五元组：

$$
S = (q_{ai}, q_{pl}, m_{ai}, m_{pl}, r)
$$

其中：

- $q_{ai}, q_{pl} \in \{0,1,\dots, Q_{\max}\}$ 为双方当前的气数。

- $m_{ai}, m_{pl} \in \{\emptyset\} \cup \mathcal{M}$ 为双方上一回合使用的招式，$\emptyset$ 表示初始回合无上一招。

- $r \in \mathbb{N}$ 为当前回合数（0-indexed）。

  

  招式空间 $\mathcal{M}$ 包含有限个动作。当启用 `反弹` 机制时，$\mathcal{M}$ 增加一个 `反弹` 动作，游戏在两种模式下分别独立学习。每个招式 $a \in \mathcal{M}$ 关联属性三元组 $\textbf{Attr}(a) = \langle \textbf{cost}(a), \textbf{atk}(a), \textbf{def}(a) \rangle$。判定函数 $J: \mathcal{M} \times \mathcal{M} \to \{-1,0,1\}$ 定义为：

  

$$
J(a_i, a_j) = 
\begin{cases}
1, & \text{if } \textbf{atk}(a_i) > \textbf{def}(a_j) \text{ and no special exemption}, \\
-1, & \text{if } \textbf{atk}(a_j) > \textbf{def}(a_i) \text{ and no special exemption}, \\
0, & \text{otherwise (draw, continue)}.
\end{cases}
$$

`跳` 和 `反弹` 的特殊规则需要专门进行处理。

### 状态转移

给定状态 $S_t$ 和双方动作 $a_{ai}, a_{pl}$，若 $J(a_{ai}, a_{pl}) \neq 0$，游戏终止，收益为该值（正表示 AI 胜，负表示玩家胜）。否则转移至下一状态：

$$
S_{t+1} = \bigl( q'_{ai},\, q'_{pl},\, a_{ai},\, a_{pl},\, r+1 \bigr)
$$

其中

$$
q' = \max\!\bigl(0,\; \min(q - \textbf{cost}(a) + \mathbb{I}_\text{bre},\; Q_{\max})\bigr)
$$

`bre` 是指 `吐气`。可以看出转移是确定性的，因此博弈树无随机性，仅策略随机。

### 博弈图构建

训练前通过 `BFS` 生成所有可达状态（DAG）。每个节点存储：

- `valid_ai` / `valid_pl`：合法招式列表（字符串）

- `transitions`：字典 `{(ma, mb): outcome}`，其中 outcome 为终局（`('terminal', w)`）或后继状态。

  为加速 CFR 迭代，在 `cfr.py` 中进一步编译为 **数值化紧凑结构** `_CompiledState`：

- `va`, `vb`：招式索引列表（整数）

- `trans_winner`：$n_a \times n_b$ 矩阵，终局时直接存储收益 $w \in \{-1,0,1\}$，否则为 $0$。

- `trans_next`：$n_a \times n_b$ 矩阵，存储后继状态哈希（整数）。

## 基于 CFR 的均衡求解

### 同时行动博弈

双方在每一轮无法观测对方出招后再决策，因此传统的 `Minimax` 算法无法直接应用。采用 **反事实遗憾最小化（CFR）** 来逼近纳什均衡，即通过反复自对弈，维护每个信息集（此处为每个状态）下每个动作的累积遗憾，并利用 `Regret Matching` 更新策略，最终平均策略收敛到近似均衡。

训练时，按回合逆序遍历所有 DAG。对每个非终止状态 $S$，定义 AI 与玩家的当前策略 $\sigma_\text{ai}, \sigma_\text{pl}$。从累积遗憾 $R_\text{ai}(S, a)$ 计算当前策略：

$$
\sigma_\text{ai}(S, a) = \frac{\max(0, R_\text{ai}(S, a))}{\sum_{a' \in \mathcal{V}_\text{ai}(S)} \max(0, R_\text{ai}(S, a'))}
$$

其中 $\mathcal{V}_\text{ai}(S)$ 为 AI 在当前状态下的合法动作集合。若所有遗憾非正，则回退到均匀分布。玩家策略 $\sigma_\text{pl}$ 类似计算。

状态 $S$ 的价值定义为双方在当前策略下的期望贴现收益：

$$
V(S) = \sum_{i \in \mathcal{V}_\text{ai}} \sum_{j \in \mathcal{V}_\text{pl}} \sigma_\text{ai}(a_i) \,\sigma_\text{pl}(a_j) \; \gamma^{r} \cdot J(a_i, a_j)
$$

其中 $\gamma \in (0,1]$ 为贴现因子，越小越偏好速胜；$r$ 为当前回合数。对于非终局转移，$J = 0$ 时需递归计算后继状态的价值：

$$
V(S) = \sum_{i, j} \sigma_\text{ai}(a_i) \sigma_\text{pl}(a_j) \, \gamma^{r} \cdot \bigl[ J(a_i, a_j) + \mathbb{I}_{\{J = 0\}} \cdot V(T(S, a_i, a_j)) \bigr]
$$

实际实现中，由于 DAG 逆序计算，后继状态的价值已被预先计算。如此，对于 AI，每个动作 $a_i$ 的反事实价值为：

$$
\text{CFV}_\text{ai}(S, a_i) = \sum_{j} \sigma_\text{pl}(a_j) \, \gamma^{r} \cdot \bigl[ J(a_i, a_j) + \mathbb{I}_{\{J = 0\}} \cdot V(T(S, a_i, a_j)) \bigr]
$$

玩家的反事实价值类似定义（注意零和对称性）。遗憾更新为：

$$
\begin{cases}
R_\text{ai}(S, a_i) \mathrel{+}= \text{CFV}_\text{ai}(S, a_i) - V(S)\\
R_\text{pl}(S, a_j) \mathrel{+}= V(S) - \text{CFV}_\text{pl}(S, a_j)
\end{cases}
$$

每次迭代产生的策略 $\sigma^t$ 被累积，并用于最终平均策略：

$$
\Sigma_{ai}(S, a) \mathrel{+}= w_t \cdot \sigma^t_{ai}(S, a)
$$

$$
\bar{\sigma}_{ai}(S, a) = \frac{C_\text{ai}(S, a)}{\displaystyle\sum_{a'} C_\text{ai}(S, a')}
$$

其中 $w_t$ 为迭代权重。标准 CFR 采用等权重，DCFR 采用线性权重即 $w_t = t$。经过 $T$ 次迭代，平均策略的遗憾上界为 $O(\displaystyle\frac{1}{\sqrt{T}})$，从而近似纳什均衡。

## 加速收敛

DCFR 对正遗憾和负遗憾施加不同的折扣因子，以加速收敛并减少振荡。设第 $t$ 次迭代后的累积遗憾为 $\bar{R}^{t}(a)$，更新方式为：

$$
\bar{R}^{t}(a) = \alpha_{+}^{t} \cdot \bar{R}^{t-1}(a) + \Delta R^{t}(a)
$$

其中折扣因子依赖于遗憾符号：

$$
\alpha_{+}^{t} = \left(\frac{t}{t+1}\right)^{p_{+}},\qquad
\alpha_{-}^{t} = \left(\frac{t}{t+1}\right)^{p_{-}}
$$

且当 $\Delta R^{t}(a) \ge 0$ 时使用 $\alpha_{+}$，否则使用 $\alpha_{-}$。典型设置取 $p_{+} > 1$（强力折扣正遗憾），$p_{-} = 0$（不折扣负遗憾，保留长期知识）。同时，累积策略采用线性权重 $w_t = t$，使后期迭代在平均策略中占据更大权重。折扣系数始终随迭代增加而衰减，后期正遗憾影响减小，策略趋于稳定。

## 自适应层

纯 CFR 给出的是稳健的纳什均衡策略，最小化可被对手利用的程度，但缺乏主动削弱对手固定模式的能力。自适应层通过在线学习对手行为（跨局与局内）调整策略，实现“读人”。自适应层通过跨局和局内学习，动态调整策略以削弱对手的模式。

### 跨局学习

按状态记录对手的历史动作频率。对每个状态 $S$，使用加法平滑估计对手的条件分布：

$$
\hat{P}_\text{pl}(m \mid S) = \frac{\textbf{count}(S, m) + \alpha}{\sum_{m' \in \mathcal{V}_\text{pl}(S)} \textbf{count}(S, m') + \alpha \cdot |\mathcal{V}_\text{pl}(S)|}
$$

其中 $\alpha$ 为平滑参数。基于此分布计算 AI 每个动作的最佳响应期望回报：

$$
\mathrm{BR}(a_i) = \sum_{j \in \mathcal{V}_\text{pl}(S)} \hat{P}_\text{pl}(m_j \mid S) \cdot J(a_i, m_j)
$$

通过 `Softmax` 将回报转换为动作分布：

$$
\sigma_{br}(a_i) = \frac{\exp\bigl( (\mathrm{BR}(a_i) - \max_{k} \mathrm{BR}(a_k)) / \tau \bigr)}{\sum_{k'} \exp\bigl( (\mathrm{BR}(a_{k'}) - \max_{k} \mathrm{BR}(a_k)) / \tau \bigr)}
$$

其中 $\tau$ 为温度参数。最终策略为 CFR 基线 $\sigma_{cfr}$ 与最优响应分布 $\sigma_{br}$ 的凸组合：

$$
\sigma_{mid}(a_i) = (1 - \lambda) \cdot \sigma_{cfr}(a_i) + \lambda \cdot \sigma_{br}(a_i)
$$

混合率 $\lambda$ 由该状态下观察次数 $n$ 决定：

$$
\lambda = \lambda_{\text{cross}-\max} \cdot \frac{n}{n + k}
$$

其中 $\lambda_{\text{cross}-\max}$ 为跨局最大混合率，$k$ 为信任阈值。观察次数越多，越信任对手模型。

### 局内学习

与跨局不同，局内学习假设对手在 **当前对局** 中可能遵循一个未知但固定的策略，从先验分布开始，随回合更新。假设对手在当前对局中采用一个固定的（未知）分布，以 **Dirichlet 先验** 建模。先验视各动作为等可能，强度由参数 $\beta$ 控制；每局开始时清空计数，随着对局推进，后验分布为：

$$
P_{post}(m) = \frac{\displaystyle\frac{\beta}{|\mathcal{V}_{pl}|} + C(m)}{\beta + T_{m}}
$$

其中 \[T_{m}\] 为当前对局中该状态已观察到的总动作数（实际上局内学习不区分状态，使用全局统计）。基于后验分布计算最佳响应和 `Softmax` 分布（与跨局类似）得到 $\sigma_{br}^{within}$。然后与当前策略 $\sigma_{mid}$（跨局修正后的）进行第二次混合：

$$
\sigma_{final}(a_i) = (1 - \lambda_{within}) \cdot \sigma_{mid}(a_i) + \lambda_{within} \cdot \sigma_{br}^{within}(a_i)
$$

其中 $\lambda_{within} = \lambda_{\text{within}-\max} \cdot \displaystyle\frac{T_{m}}{T_{m} + \beta}$，即随观测增加而增加。\[\lambda_{\text{within}-\max}\] 是局内最大混合率。


### 融合策略

级联顺序为先跨局学习，后局内学习：

$$
\sigma_{cfr} \xrightarrow{\text{cross}} \sigma_{mid} \xrightarrow{\text{within}} \sigma_{final}
$$

跨局提供长期风格记忆，局内捕捉当前对局的即时趋势。实验表明融合永远优于单一模态。

## 性格系统

性格通过调节自适应层的参数以及施加动作偏差，来改变 AI 的行为风格。核心参数包括：

- 温度参数 $\tau_{cross}, \tau_{within}$：控制 `Softmax` 的锐度，低温度使策略更贪婪（更倾向于利用观察到的最佳响应），高温度下回更具有探索性。

- 混合率参数 $\lambda_{\max}$ 及信任阈值 $k, \beta$：控制自适应层的侵略性。

- 偏差向量 $b(a)$：对最终策略施加乘法修正，用于鼓励或抑制特定类型的动作。

  修正后的策略为：

$$
\sigma_{final, biased}(a) = \frac{\sigma_{final}(a) \cdot (1 + b(a))}{\sum_{a'} \sigma_{final}(a') \cdot (1 + b(a'))}
$$

通过组合不同取值的参数集，可生成不同个性。所有性格都建立在 CFR 均衡基础之上，只做语义层面的修正，不破坏纳什均衡的整体结构。

# 代码

## 项目结构

```
/
├── launcher.py        		# 主入口
├── README.md           	# 项目说明文档
├── src/                    # 核心源代码目录
│   ├── game.py             # 游戏引擎（纯函数式逻辑，无状态）
│   ├── cfr.py              # CFR/DCFR 算法实现（离线训练）
│   ├── adaptive.py         # 自适应系统（实时策略调整）
│   ├── personality.py      # 性格系统（决策风格控制）
│   └── db.py               # SQLite 数据库层（策略持久化）
├── tests/                  # 测试目录
│   ├── validate.py         # 策略验证与评估
│   └── selfplay_test.py    # AI 自对弈测试
└── data/                   # 数据存储
    └── courage.db          # SQLite 数据库文件
```

## 使用方式

```bash
# 1. 训练策略（首次使用必须运行）
python -m tests.validate

# 2. 启动人机对战
python launcher.py

# 3. 自对抗性能测试
python -m tests.selfplay_test
```

**依赖**：Python 3.8+，标准库（无需第三方包）。

