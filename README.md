# SVDCompression
A naive image compression technique using low-rank matrix approximations obtained from the SVD.

## Background
The basic idea is that we represent the image as a matrix and compress it using a low-rank approximation.

Suppose $\boldsymbol A \in \mathbb R^{m\times n}$ is a matrix represention of an $m$ by $n$ image (or at least one colour channel of the image) and let $\boldsymbol A_k$ denote its rank $k$ approximation. The problem here is to find the matrix $\boldsymbol A_k$ such that,

$$
\Vert \boldsymbol A - \boldsymbol A_k \Vert \quad\text{is minimised with repsect to the constraint that}\quad \text{rank}(\boldsymbol A_k) \leq k
$$

It turns out that the solution to this problem when using both the matrix Frobenius norm and the spectral norm is related to the Singular Value Decomposition (SVD) of $\boldsymbol A$. This is described by the [Eckart–Young–Mirsky](https://en.wikipedia.org/wiki/Low-rank_approximation) theorem. 


Suppose $\boldsymbol A$ has an SVD given by,

$$
\boldsymbol A = \boldsymbol U \boldsymbol \Sigma \boldsymbol V^T
$$

(which always exists for any matrix)

The Eckart-Young-Mirsky theorem then states that the best rank $k$ approximation of $\boldsymbol A$ (denoted $\boldsymbol A_k$) can be computed as follows,

$$
\boldsymbol A_k = \sum_{i=1}^k \sigma_i \boldsymbol u_i \boldsymbol v_i^T
$$

where,
- $\sigma_i$ is the ith singular value (coming from $\boldsymbol \Sigma$)
- $\boldsymbol u_i$ is the ith left singular vector (coming from $\boldsymbol U)$
- $\boldsymbol v_i^T$ is the ith right singular vector (coming from $\boldsymbol V^T$)

Essentially it is saying that the best rank $k$ approximation of $\boldsymbol A$ is equal to the sum of the outer product of the first $k$ singular vectors scaled by the corresponding singular value. Each outerproduct is a rank $1$ matrix and so the final results will be of rank $k$.

## Storage Reduction
As with above suppose that the image $\boldsymbol A$ is of dimensions $m$ by $n$. This means that for each colour channel of the image you would need to store $m\times n$ pixel values. 

Now suppose that you compute the rank $k$ approximation of the image where $k \leq \text{rank}(\boldsymbol A)$. To store this in memory we need to store:
- $km$ values for the first $k$ columns of $\boldsymbol U$
- $kn$ values for the first $k$ rows of $\boldsymbol V^T$
- $k$ singular values

And so the total storage requirement is given by,

$$
k(m + n + 1)
$$

For $k$ sufficiently smaller than $\text{rank}(\boldsymbol A)$ this will be a significant reduction in storage size (at the cost however of a loss in detail). 

## Implementation
The core of the compression method can be performed in just a few lines of code thanks to NumPy,
```python
U, Sigma, Vh = np.linalg.svd(image)  # compute the SVD decomposition
compressed = U[:, :k] @ np.diag(Sigma[:k]) @ Vh[:k, :]  # find rank k approximation
```
In the program I then compress the image using several different choices for $k$ to visualise the effect on quality. Whilst computing the SVD can be quite expensive computationally, it only needs to be done once to generate all low rank approximations. Furthermore the compressed image does not need to be recalculated from scratch each time since to increase detail one only need to continue adding rank $1$ matricies instead of going back to the start and recomputing the matrix product. 
