#' MixedDistributionError
#'
#' Calculates a fitting error as well as the overlapping measure for the mixtures. Combines them with ratio rho in favor of Overlapping.
#'
#' @param Means        Means of the GMM Components
#' @param SDs              Standard Deviations of the GMM Components
#' @param Weights     Weights of the GMM Components
#' @param Data  Empirical Data based on which the GMM is build
#' @param rho   Ratio of OverlappingError vs Fitting Error
#' @param breaks   vector containing the breaks between bins
#' @param Kernels  positions at which density is to be compared
#' @param ErrorMethod   "pdeerror": fitting error is measured through Pareto Density Estimation.
#' "chisquare": fitting error is measured through the Chi Square fitting error.
#' @return Mixed Error
#' @author Florian Lerch
#'
#' @import AdaptGauss
#' @export
#' @examples
#' Data = c(rnorm(50,1,2), rnorm(50,3,4))
#' MixedDistributionError(c(1,3), c(2,4), c(0.5,0.5), Data = Data)

MixedDistributionError <- function(Means, SDs, Weights, Data, rho = 0.5, breaks = NULL, Kernels = NULL, ErrorMethod = "chisquare"){
  if(!is.vector(Data)) stop("Data is not a vector")

  NoBins = OptimalNoBins(Data)

  if(is.null(breaks)) breaks = seq(min(Data),max(Data), length.out=length(NoBins)+1)
  if(is.null(Kernels)){
    V = ParetoDensityEstimation(Data)
    kernels = V$kernels
    paretoDensity = V$paretoDensity
  }

  if(ErrorMethod == "pdeerror"){
    V = Pdf4Mixtures(kernels, Means, SDs, Weights)
    SimilarityError = sum(abs(V$PDFmixture - paretoDensity))
  }
  else if(ErrorMethod == "chisquare"){
    estimatedBins = BinProb4Mixtures(Means, SDs, Weights, breaks)*length(Data)
    norm = estimatedBins
    norm[norm<1]=1
    observedBins = hist(Data, breaks=breaks, plot = F)$counts
    diffssq = (observedBins - estimatedBins)^2
    diffssq[diffssq<4] = 0 # siehe Chi2testMixtures. Weniger als 2^2 diff, ist identisch
    SimilarityError = sum(diffssq/norm)

    SimilarityError = SimilarityError / length(Data)
  }
  else{
    stop(paste("ErrorMethod", ErrorMethod, "is not recognized. Please use either 'pdeeerror' or 'chisquare'"))
  }

  OError = OverlapErrorByDensity(Means, SDs, Weights, Data, Kernels)$OverlapError
  DistributionError = rho*SimilarityError + (1-rho)*OError

  return(list(SimilarityError = SimilarityError, OverlapError = OError, MixedDistributionError = DistributionError))
}
