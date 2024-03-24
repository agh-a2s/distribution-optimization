#' OverlapErrorByDensity
#'
#' Similarity in GMM by Density
#'
#' Calculates the similarity (overlap) between multiple modes in Gaussian Mixture Models. Kernels at equally distanced positions
#' are used, if not explicitly given.
#'
#' @param Means        Means of the GMM Components
#' @param SDs              Standard Deviations of the GMM Components
#' @param Weights     Weights of the GMM Components
#' @param Data  Dataset that the GMM should be compared with
#' @param Kernels  if length(Kernels) = 1: amount of kernels
#'             if length(Kernels) > 1: kernels in dataspace at which the GMM Components will be compared with each other
#' @return List:
#' OverlapError    Error for estimating the maximal Overlap of AUC of PDFs of each pair of GMM Components
#' Kernels         Kernels that were used for comparing the GMM Components
#' @author Florian Lerch
#' @importFrom AdaptGauss OptimalNoBins
#' @export
#' @examples
#' Data = c(rnorm(50,1,2), rnorm(50,3,4))
#' V<-OverlapErrorByDensity(c(1,3), c(2,4), c(0.5,0.5), Data = Data)
#' AdaptGauss::PlotMixtures(Data, c(1,3), c(2,4), SingleGausses = TRUE)
#' print(V$OverlapError)
OverlapErrorByDensity <- function(Means, SDs, Weights, Data=NULL, Kernels=NULL){
  #
  # OverlapError <- OverlapErrorByDensity(Means, SDs, Weights, Data, Kernels)
  # INPUT
  # Means[1:m]     Means of the GMM Components
  # SDs[1:m]       Standard Deviations of the GMM Components
  # Weights[1:m]   Weights of the GMM Components
  # OPTIONAL
  # Data[1:n]      Dataset that the GMM should be compared with
  # Kernels[1:c]   if length(Kernels) = 1: amount of kernels
  #                if length(Kernels) > 1: kernels in dataspace at which the GMM Components will be compared with each other
  # OUTPUT
  # OverlapError    Error for estimating the maximal Overlap of AUC of PDFs of each pair of GMM Components
  # Kernels         Kernels that were used for comparing the GMM Components
  # author: FL

  errorOverlapComponent = NULL

  if(is.null(Data)){
    if(is.null(Kernels)) stop("OverlapErrorDensity: either Data or Kernels needs to be given")
    if(length(Kernels == 1)) stop("OverlapErrorDensity: if Data is not given, Kernels need to specify positions in the dataspace in form of a vector")
  }

  if(is.null(Kernels)) Kernels = OptimalNoBins(Data)
  if(length(Kernels) == 1) Kernels = seq(min(Data), max(Data), length.out = Kernels)

  # lade densities fuer alle Moden an allen Positionen (Kernels)
  densities = matrix(nrow = length(Means), ncol=length(Kernels))
  for(i in 1:length(Means)){
    densities[i,] = dnorm(Kernels, Means[i], SDs[i]) * Weights[i]
  }

  tryCatch({
    overlapInComponent = matrix(nrow=length(Means), ncol=length(Kernels))
    for(i in 1:length(Means)){
      overlapInComponent[i,] = apply(densities[-i,,drop=FALSE], 2, max) # jeweils hoechste mode ausser i
      oversize = overlapInComponent[i,] > densities[i,]
      overlapInComponent[i,oversize] = densities[i, oversize]
    }
  }, error=function(e) browser())

  areaInComponent = rowSums(densities)
  overlapInComponents2 = rowSums(overlapInComponent)
  ovRatioInComponent = overlapInComponents2 / areaInComponent

  errorOverlapComponent = max(ovRatioInComponent)
  if(is.na(errorOverlapComponent))
    errorOverlapComponent = 0

  return(list(OverlapError = errorOverlapComponent, Kernels = Kernels))
}
