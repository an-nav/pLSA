# pLSA(Probabilistic latent semantic analysis)

## 模型对于文档生成的认知

pLSA模型认为生成文档的过程是这样的：作者以一定的概率$P(d_i)$选中了一篇文档，而后在选中文档$d_i$的前提下以$P(z_k|d_i)$的概率选中了文档的**隐含主题** $z_k$，而后在主题之下存在词分布每次以$P(w_j|z_k)$的概率生成一个词从而生成了一篇文档。

将上述描述转换成概率图模型即如下图：

<img src="D:\Data analysis\Tools of Data Analysis\Tools-and-Knowledge-of-Data-Analysis\Resource\pLSA_1.png" alt="pLSA概率模型图" style="zoom:40%;" />

从图中可以看到文档$d_i$生成了文档主题$z_k$而文档主题$z_k$生成了文档中的词$w_j$,图中的方框表示重复这个过程即每篇文档有N个词，一共M篇文档。显然对于文档$d_i$和词$w_j$是可以观测的而我们不知道的是文档主题$z_k$是如何分布的因此我们模型要学习的就是这个$z_k$。接下我们就根据这个概率图模型来看看如何求解。

## 模型的求解

### 符号定义

首先说明一下推导中会用到的一些符号：

观测变量为：
$文档\,d_i \in D(共有M篇文档)$
$词\, w_j \in W(共有V个互异的词)$
隐变量为:
$主题\,z_k\in Z(共有K个主题)$
概率:
$P(d_i)表示选中文档d_i的概率$
$P(z_k|d_i)表示在给定文档d_i下选中主题z_k的概率$
$P(w_j|z_k)表示在给定主题z_k下选中词w_j的概率$
$P(w_j|d_i)表示在给定文档d_i下选中从w_j的概率$

结合上面的概率图，pLSA模型的参数$\theta$显然是$P(z_k|d_i)与P(w_j|z_k)$即$\theta=(P(z_k|d_i),P(w_j|z_k))$

所以我们需要求解的就是$K\times M个P(z_k|d_i)和V\times K个P(w_j|z_k)$。

### 推导求解

#### 联合概率

根据概率图可知联合概率:
$$
\begin{align}
P(d_i,w_j,z_k)&=P(d_i)P(z_k|d_m)P(w_j|z_k)\tag{1}\\
P(w_j,d_i)&=P(d_i)P(w_j|d_i)\tag{2}\\
由此生成一&篇文档d_i 的概率为：\\
P(\overrightarrow{w}|d_i)&=\prod_{j=1}^NP(w_j|d_i)\tag{3}\\
\,\\
\rightarrow P(w_j|d_i)&=\sum_k P(z_k|d_i)P(w_j|z_k,d_i)\\
见概率图中d_i和w_j的关&系就是概率图中典型的head-tail关系\\
\therefore P(w_j|d_i)&=\sum_k P(z_k|d_i)P(w_j|z_k)\\
\therefore P(w_j,d_i)&=P(d_i)\sum_k P(z_k|d_i)P(w_j|z_k)\tag{4}\\
\end{align}
$$

#### 似然函数

似然函数：
$$
\begin {align}
L(\theta)&=ln[\prod_{i=1}^M\prod_{j=1}^NP(d_i,w_j)^{n(w_j,d_i)}]\\
&=\sum_{i=1}^M\sum_{j=1}^Nn(w_j,d_i)lnP(d_i,w_j)\\
&=\sum_{i=1}^M\sum_{j=1}^Nn(w_j,d_i)[lnP(d_i)+ln(\sum_kP(z_k|d_i)P(w_j|z_k))]\\
&=\sum_{i=1}^M\sum_{j=1}^Nn(w_j,d_i)ln(\sum_kP(z_k|d_i)P(w_j|z_k))+\sum_{i=1}^M\sum_{j=1}^Nn(w_j,d_i)lnP(d_i)\\
\because  \sum_{i=1}^M&\sum_{j=1}^Nn(w_j,d_i)lnP(d_i)与\theta无关\\
\therefore L(\theta)&=\sum_{i=1}^M\sum_{j=1}^Nn(w_j,d_i)ln(\sum_kP(z_k|d_i)P(w_j|z_k))\tag{5}\\
\theta&=(P(z_k|d_i),P(w_j|z_k))
\end {align}
$$

式中$n(d_i)$表示文档$d_i$中的词的个数$n(w_j,d_i)$表示词$w_j$在文档$d_i$中出现的频率显然有$\sum_{j}n(w_j,d_i)=n(d_i)$。

#### EM算法

具体原理可参考另一篇EM算法文档

##### E-step

对于EM算法有Q函数:
$$
Q(\theta,\theta^{(t)})=\sum_kP(Z|X;\theta^{(t)})lnP(X,Z;\theta)
$$
其中Z为隐变量，X为观察变量，$\theta^{(t)}$为t时刻步的参数值，$\theta$为所要估计的参数变量。

将pLSA模型带入E-step得Q函数为:
$$
\begin{align}
Q(\theta,\theta^{(t)})&=\sum_i\sum_jn(d_i,w_j)E_{z_k|w_j,d_i;\theta^{(t)}}[lnP(w_j,z_k|d_i)]\\
&=\sum_i\sum_jn(d_i,w_j)\sum_kP(z_k|w_j,d_i;\theta^{(t)})lnP(w_j,z_k|d_i)\tag{6}\\
\end{align}
$$
其中：
$$
\begin{align}
&P(w_j,z_k|d_i)=P(w_j|z_k,d_i)P(z_k|d_i)=P(w_j|z_k)P(z_k|d_i)\tag{7}\\
\end{align}
$$

$$
\begin{align}
P(z_k|w_j,d_i;\theta^{(t)})&=\frac{P(w_j,d_i,z_K;\theta^{(t)})}{P(w_j,d_i;\theta^{(t)})}\\
&=\frac{P(d_i;\theta^{(t)})P(z_k|d_i;\theta^{(t)})P(w_j|z_k;\theta^{(t)})}{P(d_i;\theta^{(t)})\sum_m P(z_m|d_i;\theta^{(t)})P(w_j|z_m;\theta^{(t)})}\\
&=\frac{P(z_k|d_i;\theta^{(t)})P(w_j|z_k;\theta^{(t)})}{\sum_m P(z_m|d_i;\theta^{(t)})P(w_j|z_m;\theta^{(t)})}\tag{8}
\end{align}
$$

将式子(7)与式子(8)带入式(6)的Q函数为:
$$
Q(\theta,\theta^{(t)})=\sum_i\sum_jn(d_i,w_j)\sum_K\frac{P(z_k|d_i;\theta^{(t)})P(w_j|z_k;\theta^{(t)})}{\sum_m P(z_m|d_i;\theta^{(t)})P(w_j|z_m;\theta^{(t)})}ln(P(w_j|z_k)P(z_k|d_i))\tag{9}
$$

##### M_step

最大化Q函数得到$\theta^{(t+1)}$：
$$
\begin{align}
&Max\quad\theta^{(t+1)}=\mathop{argmax}_\theta \,Q(\theta,\theta^{(t)})\\
&s.t.\quad\left \{ 
\begin{array}{c}
\sum_jP(w_j|z_k)=1 \\ 
\sum_kP(z_k|d_i)=1 \\
\end{array}
\right.
\end{align}
$$
用拉格朗日乘数法求解，引入乘子$\tau_k和\rho_i$的函数H为：
$$
H=Q(\theta,\theta^{(t)})+\sum_k\tau_k(1-\sum_jP(w_j|z_k))+\sum_i\rho_i(1-\sum_kP(z_k|d_i))\tag{10}
$$
对参数求偏导并另其等于0得：
$$
\begin{align}
\frac{\partial}{\partial P(w_j|z_k)}&=\frac{\sum_in(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})}{P(w_j|z_k)}-\tau_k=0\\
\rightarrow P(w_j|z_k)&=\frac{\sum_in(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})}{\tau_k}\\
\frac{\partial}{\partial P(z_k|d_i)}&=\frac{\sum_jn(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})}{P(w_j|z_k)}-\tau_k=0\\
\rightarrow  P(z_k|d_i)&=\frac{\sum_jn(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})}{\rho_i}\\
\end{align}
$$
又因为约束条件可将乘子消去:
$$
\begin{align}
\sum_jP(w_j|z_k)&=\sum_j\frac{\sum_in(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})}{\tau_k}=1\\
\rightarrow \tau_k&=\sum_j\sum_in(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})\\
\sum_kP(z_k|d_i)&=\sum_k\frac{\sum_jn(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})}{\rho_i}=1\\
\rightarrow \rho_i&=\sum_k\sum_jn(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})

\end{align}
$$
最后得到t+1时刻参数:
$$
\begin{align}
P(w_j|z_k)^{(t+1)}&=\frac{\sum_in(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})}{\sum_j\sum_in(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})}\\
P(z_k|d_i)^{(t+1)}&=\frac{\sum_jn(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})}{\sum_k\sum_jn(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})}\\
&=\frac{\sum_jn(d_i,w_j)P(z_k|d_i,w_j;\theta^{(t)})}{n(d_i)}\\
\rightarrow \theta^{(t+1)}&=(P(w_j|z_k)^{(t+1)},P(z_k|d_i)^{(t+1)})\tag{11}
\end{align}
$$
不断迭代E-step与M-step即可解得pLSA模型的解。

## 参考

1. [博客园-NLP —— 图模型（三）pLSA（Probabilistic latent semantic analysis，概率隐性语义分析）模型](https://www.cnblogs.com/Determined22/p/7237111.html)

2. [CSDN-从贝叶斯方法谈到贝叶斯网络](https://blog.csdn.net/v_july_v/article/details/40984699)

3. [CSDN-通俗理解LDA主题模型](https://blog.csdn.net/v_JULY_v/article/details/41209515)

















