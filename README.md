# SVDCompression
A naive image compression technique using low-rank matrix approximations obtained from the SVD.

## Background
The basic idea is that we represent the image as a matrix and compress it using a low-rank approximation. This is achieved using the [Eckart–Young–Mirsky](https://en.wikipedia.org/wiki/Low-rank_approximation) theorem. 

Suppose $\boldsymbol A \in \mathbb R^{m\times n}$ is a matrix represention of an $m$ by $n$ image (or at least one colour channel of the image). Let its Singular Value Decomposition (SVD) be given by,

$$
\boldsymbol A = \boldsymbol U \boldsymbol \Sigma \boldsymbol V^T
$$

(which always exists for any matrix)

The Eckart-Young-Mirsky theorem then states that the best rank $k$ approximation of $\boldsymbol A$ (denoted $\boldsymbol A_k$) can be computed as follows,

$$
\boldsymbol A_k = \sum_{i=1}^k \sigma_i \boldsymbol u_i \boldsymbol v_i^T
$$

d
