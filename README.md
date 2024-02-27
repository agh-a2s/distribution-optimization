# distribution-optimization-py: Distribution Optimization in Python

The Gaussian Mixture Model (GMM) is rephrased as an optimization challenge and tackled through a Hierarchic Memetic Strategy. This approach diverges from the traditional Likelihood Maximization, Expectation Maximization solution. Instead, it introduces a unique fitness function that combines the chi-square test, used for analyzing distributions, with an innovative metric designed to estimate the overlapping area under the curves among various Gaussian distributions. This library offers an implementation of [DistributionOptimization](https://cran.r-project.org/web/packages/DistributionOptimization/index.html) in Python with few improvements.

### Relevant literature
- Lerch, F., Ultsch, A. & Lötsch, J. Distribution Optimization: An evolutionary algorithm to separate Gaussian mixtures. Sci Rep 10, 648 (2020). doi: [10.1038/s41598-020-57432-w](https://doi.org/10.1038/s41598-020-57432-w)
