#' Bin Probabilities
#'
#' Calculates the probability of bins/intervals within the dataspace defined by given breaks between them.
#'
#' @param Means        Means of the GMM Components
#' @param SDs              Standard Deviations of the GMM Components
#' @param Weights     Weights of the GMM Components
#' @param Breaks  Breaks Defining c-1 or c+1 bins (depending on LimitsAreFinite)
#' @param IsLogDistribution   If True, the GMM is interpreted as a logarithmic
#' @param LimitsAreFinite    If True, there are c+1 Bins, where the first and last bin are of inifinite size
#' @return Probabalities of either c-1 or c+1 bins/intervals (depending on LimitsAreFinite)
#' @author Florian Lerch
#' @export
#' @examples  
#' Data = c(rnorm(50,1,2), rnorm(50,3,4))
#' NoBins = AdaptGauss::OptimalNoBins(Data)
#' breaks = seq(min(Data),max(Data), length.out=length(NoBins)+1)
#' BinProb4Mixtures(c(1,3), c(2,4), c(0.5,0.5), breaks)
BinProb4Mixtures <- function(Means, SDs, Weights, Breaks, IsLogDistribution = rep(F, length(Means)), LimitsAreFinite=T){
# calculates the probability of bins/intervals within the dataspace defined by the given breaks between them
# BinProbabilities <- BinProb4Mixtures(Means, SDs, Weights, Breaks)
# INPUT
# Means[1:m]          Means of the GMM Components
# SDs[1:m]            Standard Deviations of the GMM Components
# Weights[1:m]        Weights of the GMM Components
# Breaks[1:c]         Breaks Defining c-1 or c+1 bins (depending on LimitsAreFinite)
# IsLogDistribution   If True, the GMM is interpreted as a logarithmic
# LimitsAreFinite     If True, there are c+1 Bins, where the first and last bin are of inifinite size
# OUTPUT
# BinProbabilities    Probabalities of either c-1 or c+1 bins/intervals (depending on LimitsAreFinite)
# Author: FL


 res = CDFMixtures(Breaks,Means = Means, SDs = SDs, Weights = Weights, IsLogDistribution = IsLogDistribution)
 cdfs = res$CDFGaussMixture #/ length(Means)
 #cdfsSingle =  res$CDFSingleGaussian

 if(!LimitsAreFinite) cdfs = c(0, cdfs, 1)

 lower = cdfs[1:(length(cdfs)-1)]
 upper = cdfs[2:(length(cdfs))]
 BinProbabilities = (upper-lower)

 return(BinProbabilities = BinProbabilities)
}
