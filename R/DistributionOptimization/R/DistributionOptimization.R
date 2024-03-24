#' Distribution Fitting
#'
#' Fits a Gaussian Mixture Model onto a Dataset by minimizing a fitting error through evolutionary optimization. Every individual encodes one GMM.
#' Details over the evolutionary process itself can be taken from the 'GA' package. \link[GA]{ga}
#'
#' @param Data           Data to be modelled
#' @param Modes               Number of expected Modes
#' @param Monitor     0:no monitoring, 1: status messages, 2: status messages and plots, 3: status messages, plots and calculated error-measures
#' @param SelectionMethod
#'   1: LinearRank selection
#'   4: UnbiasedTournament
#'   5: FitnessProportional selection
#' @param MutationMethod
#'   1: UniformRandom mutation
#'   2: NonuniformRandom mutation
#'   4: Focused mutation,  alternative random mutation around solution
#'   5: GaussMutationCust
#'   6: TwoPhaseMutation - mutation is uniform random during the first half of iterations, and than focuses around current solution
#' @param CrossoverMethod
#'   1: single point crossover
#'   2: whole arithmetic crossover
#'   3: local arithmetic crossover
#'   4: blend crossover
#'   5: GaussCrossover - exchange complete gaussian components
#'   6: MultiPointCrossover  - Random amount of information between mixtures get exchanged
#' @param PopulationSize Size of the population
#' @param MutationRate  amount (0..1) of population that gets mutated
#' @param Elitism amount of best individuals that will survive generation unchanged
#' @param CrossoverRate amount of individuals that will be used for crossover
#' @param Iter number of iterations of this algorithm
#' @param OverlapTolerance ratio between Chi-Square and OverlapError (only if FitnessMethod = 4 (Chi2ValueWithOverlap))
#' @param IsLogDistribution  which gauss components should be considered as log gaussian
#' @param ErrorMethod "pde": fitting is measured by pareto density estimation. "chisquare": fitting is measured by Chi-Square test
#' @param NoBins Number of Bins that will be used for evaluating fitting
#' @param Seed   Random Seed for reproducible results
#' @param ConcurrentInit  If true, before initialization a number of short optimizations are done to find a good starting point for evolution
#' @param ParetoRad   Pareto Radius for Pareto Density Estimation and its plots
#' @return The GA object containing the evolutionary training and a description of the final GMM consisting of means, sdevs and weights.
#' @author Florian Lerch
#' @author Jorn Lotsch
#' @author Alfred Ultsch
#'
#' @import GA
#' @import AdaptGauss
#' @import ggplot2
#' @importFrom graphics "hist" "lines" "par" "plot.new" "text"
#' @importFrom stats "dnorm" "ecdf" "rnorm" "runif"
#' @importFrom utils "head" "tail"
#'
#' @export
#'
#' @examples
#' \dontrun{
#' DistributionOptimization(c(rnorm(200),rnorm(200,3), 2))
#' }
DistributionOptimization <-
  function(Data,
           Modes,
           Monitor = 1,
           SelectionMethod = "UnbiasedTournament",
           MutationMethod = "Uniform+Focused",
           CrossoverMethod = "WholeArithmetic",
           PopulationSize = Modes * 3 * 25,
           MutationRate = 0.7,
           Elitism = 0.05,
           CrossoverRate = 0.2,
           Iter = Modes * 3 * 200,
           OverlapTolerance = NULL,
           IsLogDistribution = rep(F, Modes),
           ErrorMethod = "chisquare",
           NoBins = NULL,
           Seed = NULL,
           ConcurrentInit = F,
           ParetoRad = NULL,
           Budget = NULL) {
    # V = DistributionOptimization(Data, Modes, Monitor = F)
    
    # OUTPUT
    # list of:
    # GA      GA object of the training
    # Weights
    # SDs
    # Means
    # author: FL
    
    if (!is.null(Seed))
      set.seed(Seed)
    
    if (Monitor > 1)
      plot.new()
    
    # spannweite der daten
    dataRange = abs(max(Data) - min(Data))
    # Bestimmung der Min und Max Werte
    minSD = dataRange * 0.001 # 0.1 prozent der datarange
    #maxSD = dataRange*3 # 99,7 prozent der Daten liegen innerhalb dieser standardabweichung
    maxSD = dataRange * 0.1
    #maxSD =
    
    minMean = min(Data) # nicht sicher ob das wirklich immer gelten sollte
    maxMean = max(Data)
    #maxMean = maxMean * 1.3
    minWeight = 0.03
    maxWeight = 1
    minimums = c(rep(minWeight, Modes), rep(minSD, Modes), rep(minMean, Modes))
    maximums = c(rep(maxWeight, Modes), rep(maxSD, Modes), rep(maxMean, Modes))
    
    SortedGauss = T
    Optim = F
    restrictEMutation = F
    OverlapMethod = "Area"
    Means = NULL
    SDs = NULL
    Weights = NULL
    
    ## Werte die vorberechnet werden
    ##############
    # Vorberechnungen (notwendig fuer die Berechnung der Fehler (=> Fitness))
    ##############
    if (is.null(ParetoRad))
      ParetoRad = ParetoRadius(Data, maximumNrSamples = 5000)
    V = ParetoDensityEstimation(Data, paretoRadius = ParetoRad)
    kernels = V$kernels
    paretoDensity = V$paretoDensity
    if (is.null(NoBins))
      NoBins = OptimalNoBins(Data)
    breaks = seq(min(Data), max(Data), length.out = NoBins + 1)
    empiricalCdf = ecdf(Data)(kernels)
    observedBins = hist(Data, breaks = breaks, plot = F)$counts
    if (length(observedBins) != NoBins)
      stop("R Base Function hist failed and produced a different number of bins")
    
    # globale Variablen
    bestChi2Value = -Inf
    bestChi2P = Inf
    bestKS = Inf
    
    Kernels = seq(min(Data), max(Data), length.out = 40)
    
    if (is.null(OverlapTolerance)) {
      if (ErrorMethod == "chisquare")
        OverlapTolerance = 0.5
      else if (ErrorMethod == "pde")
        OverlapTolerance = 0.2
    }
    
    
    ##################################
    #### Fitness Funktion ############
    ##################################
    GaussmixtureFitness <- function(x) {
      # Fitnessfunktion zur Benutzung mit dem GA Paket
      # x ist ein Vektor mit allen Parametern
      # Rueckgabewert ist ein Fitnesswert
      
      # werte aus x auslesen
      Weights = x[1:Modes]
      SDs = x[(Modes + 1):(Modes * 2)]
      Means = x[(Modes * 2 + 1):(Modes * 3)]
      
      N = length(Data)
      
      # gewichte jeweils von 0 bis 1 muessen so umgeformt werden, dass sie in Summe eins ergeben
      Weights = Weights / sum(Weights)
      
      if (any(SDs < minSD) | any(SDs > maxSD) |
          any(Means < minMean) | any(Means > maxMean) |
          any(Weights < minWeight) | any(Weights > maxWeight)){
        res = 10
        return(-res)  
      }
      
      ####
      # calculation of similarity distribution error
      ####
      if (ErrorMethod == "chisquare") {
        estimatedBins = BinProb4Mixtures(Means, SDs, Weights, breaks) * N
        norm = estimatedBins
        norm[norm < 1] = 1
        diffssq = (observedBins - estimatedBins) ^ 2
        diffssq[diffssq < 4] = 0 # siehe Chi2testMixtures. Weniger als 2^2 diff, ist identisch
        SimilarityError = sum(diffssq / norm)
        # normierungskram
        #SimilarityError = SimilarityError * (1 / (4*(N^2))) # theoretisches maximum
        #SimilarityError = SimilarityError^(1/6) # gleichmaessigere Verteilung zwischen 0 und 1
        
        
        #SimilarityError = (SimilarityError * (1 / N))
        SimilarityError = SimilarityError / N
      }
      else if (ErrorMethod == "pde") {
        V = Pdf4Mixtures(kernels, Means, SDs, Weights)
        SimilarityError = sum(abs(V$PDFmixture - paretoDensity))
      }
      else{
        stop(paste0("Errormethod '", ErrorMethod, "' is not recognized."))
      }
      
      #res <- Chi2testMixtures(Data, Means, SDs, Weights)
      #res <- Chi2Value(Data, Means, SDs, Weights)
      
      a <- OverlapTolerance  # gewichtung chi2
      
      #if(!PDEMode){
      #  res <- Chi2ValueWithOverlap(Means, SDs, Weights, precalc, IsLogDistribution = IsLogDistribution, preciseOverlap = F,
      #                              customOverlapFunction = customOverlapFunction)
      #  res <- a*res$normedChi2Value + (1-a)*res$overlappingValue
      #}
      #browser()
      
      if (OverlapTolerance < 1) {
        #if(is.null(customOverlapFunction))
        #if(OverlapMethod == "Datapoints")
        #OError = verlapErrorC(
        #else if(OverlapMethod == "Area")
        #OError = OverlapErrorByDensity(Means, SDs, Weights, Data, Kernels)
        OError = OverlapErrorByDensity(Means, SDs, Weights, Data, Kernels)$OverlapError
        #OError = OverlapErrorByE(Means, SDs, Weights, Data)$errorAvg
        #else
        #  OError = customOverlapFunction(Means, SDs, Weights, Data, paretoDensity, kernels)
      }
      else{
        OError = 0
      }
      
      
      if (OError > OverlapTolerance)
        res = 10
      else
        res <- SimilarityError
      
      #return(res$Pvalue)
      return(-res)
    }
    
    ####################
    ### Mutation #######
    ####################
    GaussMutationWithIntensity <- function(obj, parent) {
      mutate <- parent <- as.vector(obj@population[parent,])
      dempeningFactor <- 1 - obj@iter / obj@maxiter
      
      mutateParameter <- sample(length(parent), 1)
      
      min <- obj@lower[mutateParameter]
      max <- obj@upper[mutateParameter]
      
      range <- (max - min) * dempeningFactor
      
      #  min = max(mutate[mutateParameter]-range, obj@min[mutateParameter] )
      #  max = min(mutate[mutateParameter]+range, obj@max[mutateParameter] )
      
      min = mutate[mutateParameter] - range
      max = mutate[mutateParameter] + range
      
      if (min < obj@lower[mutateParameter])
        min = obj@lower[mutateParameter]
      if (max < min)
        max = min
      
      mutate[mutateParameter] <- runif(1, min, max)
      return(mutate)
      
    }
    
    UniformMutation <- function(object, parent, ...) {
      mutate <- parent <- as.vector(object@population[parent,])
      n <- length(parent)
      
      for (i in 1:sample(1:3, 1)) {
        j <- sample(1:n, size = 1)
        
        min = object@lower[j]
        max = object@upper[j]
        
        if (restrictEMutation &&
            (j >= (Modes * 2 + 1):(Modes * 3)) &&
            (j <= (Modes * 3))) {
          # es handelt sich um einen Erwartungswert
          if (j != (Modes * 3))
            # nicht die letzte Mode
            max = mutate[j + 1]
          if (j != ((Modes * 2) + 1))
            # nicht die erste Mode
            min = mutate[j - 1]
        }
        
        mutate[j] <- runif(1, min, max)
      }
      return(mutate)
    }
    
    GaussMutationCust <- function(obj, parent) {
      mutate <- parent <- as.vector(obj@population[parent,])
      n <- length(parent)
      
      #### replace random points
      # mutation_amount = sample(1:n,1)
      
      # par <- sample(1:n, mutation_amount
      
      #### replace component
      # com = sample(1:Modes, 1)
      # par <- c(com, Modes+com, Modes*2+com)
      
      ### start with type
      # type = sample(1:3,1)
      # if(type==1) par = c(1:Modes)
      # if(type==2) par = sample((Modes+1):(Modes*2),1 )
      # if(type==3) par = sample((Modes*2+1):(Modes*3),1 )
      
      ### always resample complete type
      #type = sample(1:3,1)
      #if(type==1) par = c(1:Modes)
      #if(type==2) par = c((Modes+1):(Modes*2))
      #if(type==3) par = c((Modes*2+1):(Modes*3))
      
      ### focus sampling on ew and sd
      type = sample(1:9, 1)
      if (type <= 3)
        par = sample((Modes + 1):(Modes * 2), 1)
      else if (type <= 8)
        par = sample((Modes * 2 + 1):(Modes * 3), 1)
      else
        par = c(1:Modes)
      
      
      for (i in par) {
        mutate[i] <- runif(1, obj@lower[i], obj@upper[i])
      }
      
      return(mutate)
    }
    
    TwoPhaseMutation <- function(obj, parent) {
      if (obj@iter <= (obj@maxiter * 0.8)) {
        res <- gareal_raMutation(obj, parent)
      }
      else{
        res <- GaussMutationWithIntensity(obj, parent)
      }
      
      if (any(is.nan(res))) {
        print("error in result after two phase mutation")
        browser()
      }
      
      return(res)
    }
    
    ####################################
    ###### Crossoverfunktionen  ########
    ####################################
    GaussCrossover <- function(object, parents) {
      parents <- object@population[parents, , drop = FALSE]
      n <- ncol(parents)
      children <- matrix(as.double(NA), nrow = 2, ncol = n)
      
      # entscheide wie viele moden vom ersten elter stammen
      nrOfModes = round(runif(runif(1, min = 1, max = Modes - 1)))
      modesOfParent1 = sample(1:Modes, nrOfModes)
      r <- rep(FALSE, Modes)
      r[modesOfParent1] = TRUE
      
      for (i in 1:Modes) {
        x = c(i, i + Modes, i + (Modes * 2))
        if (r[i]) {
          children[1, x] = parents[1, x]
          children[2, x] = parents[2, x]
        }
        else{
          children[1 , x] = parents[2, x]
          children[2 , x] = parents[1, x]
        }
      }
      
      out <- list(children = children, fitness = rep(NA, 2))
      return(out)
    }
    
    MultiPointCrossover <- function(object, parents, ...) {
      fitness <- object@fitness[parents]
      parents <- object@population[parents, , drop = FALSE]
      n <- ncol(parents)
      children <- matrix(as.double(NA), nrow = 2, ncol = n)
      
      
      crossoverPoints = sample(1:n, 4)
      cP2 <- setdiff(1:n, crossoverPoints)
      
      fitnessChildren <- rep(NA, 2)
      
      children[1, crossoverPoints] <- parents[1, crossoverPoints]
      children[2, crossoverPoints] <- parents[2, crossoverPoints]
      children[1, cP2] <- parents[2, cP2]
      children[2, cP2] <- parents[1, cP2]
      
      out <- list(children = children, fitness = fitnessChildren)
      return(out)
    }
    
    #####################################
    ##### Monitoring ####################
    #####################################
    MonitorFun <- function(x) {
      if (Monitor == 0)
        return("")
      else if (Monitor == 1)
        gaMonitor(x)
      else{
        best <- which.max(x@fitness)
        bestSol <- x@population[best, ]
        
        res = bestSol
        
        Weights = res[1:Modes]
        SDs = res[(Modes + 1):(Modes * 2)]
        Means = res[(Modes * 2 + 1):(Modes * 3)]
        Weights = Weights / sum(Weights)
        
        if (!(x@iter %% 20)) {
          # nur bei jeder 20. iteration plotten
          if ((!(x@iter %% 100)) &&
              (Monitor >= 3)) {
            # nur bei jeder 100. iteration Fehler berechnen
            
            V = Chi2testMixtures(Data, Means, SDs, Weights, PlotIt = F)
            bestChi2Value <<- V$Chi2Value
            bestChi2P <<- V$Pvalue
            
            V = KStestMixtures(Data, Means, SDs, Weights)
            bestKS <<- V$PvalueKS
          }
          
          xl = max(Data, na.rm = T)
          xa = min(Data, na.rm = T)
          xlim = c(xa, xl)
          
          #        if(!is.null(ImagePath)){
          #          jpeg(paste0(ImagePath, "gmm-", x@iter, ".jpg"), width = 600, height = 600)
          #          PlotMixtures(Data, Means, SDs=SDs,Weights = Weights, SingleGausses = T, IsLogDistribution = IsLogDistribution,
          #                       ParetoRad=ParetoRad, lwd=2, xlim=xlim)
          #          dev.off()
          #        }
          
          par(mfrow = c(2, 2))
          # PDE Plot
          PlotMixtures(
            Data,
            Means,
            SDs = SDs,
            Weights = Weights,
            SingleGausses = T,
            IsLogDistribution = IsLogDistribution,
            ParetoRad = ParetoRad,
            lwd = 2,
            xlim = xlim
          )
          
          # Histogram Plot
          V1 <- hist(Data, breaks = breaks)
          X = sort(Data)
          pdfV = Pdf4Mixtures(X, Means, SDs, Weights)
          yfit <- pdfV$PDFmixture * diff(V1$mids[1:2]) * length(Data)
          lines(X, yfit, col = "red", lwd = 2)
          
          # CDF Plot
          plot(
            ecdf(Data),
            xlab = "",
            ylab = "",
            main = "CDF",
            verticals = TRUE,
            pch = 20,
            cex = 0.8,
            col = "red"
          )
          
          
          V = CDFMixtures(X, Means, SDs, Weights)
          lines(X,
                V$CDFGaussMixture,
                col = "blue",
                lwd = 2)
          
          if (Monitor >= 3) {
            # erstelle einen leeren Plot zum Schreiben
            plot(
              0,
              xaxt = 'n',
              yaxt = 'n',
              bty = 'n',
              pch = '',
              ylab = '',
              xlab = ''
            )
            
            text(paste("Chi2 Value: ", bestChi2Value),
                 y = 0.9,
                 x = 1)
            text(paste("Chi2 P-Value: ", bestChi2P),
                 y = 0.5,
                 x = 1)
            text(paste("KS P-Value: ", bestKS),
                 y = 0.2,
                 x = 1)
          }
          
          par(mfrow = c(1, 1))
          
        }
        
        
        
        # print(paste0(c("SDs:", SDs), collapse=" "))
        # print(paste0(c("Means:", Means), collapse=" "))
        # print(paste0(c("Weights:", Weights), collapse=" "))
        #
        # print(paste("Elitism:", Elitism))
        # print(paste("Population:", PopulationSize))
        
        #print(Chi2ParetoValue(Data, Means, SDs, Weights))
        
        gaMonitor(x)
      }
    }
    
    biasedTournamentSelection <-
      function(object,
               tournamentSize = 4,
               p = 0.5,
               winners = 1,
               ...) {
        sel <- rep(NA, object@popSize)
        
        i = 1
        while (i < object@popSize) {
          s <- sample(1:object@popSize, size = tournamentSize)
          #prob <- abs(object@fitness[s])/sum(abs(object@fitness[s]))
          
          r <- rank(abs(object@fitness[s]), ties.method = "random")
          prob = p * (1 - p) ^ (r - 1)
          
          tSel <- sample(
            s,
            size = 1,
            prob = pmin(pmax(0,
                             prob), 1, na.rm = TRUE),
            replace = TRUE
          )
          
          sel[i:(i + winners - 1)]
          i = i + winners
        }
        sel = sel[1:object@popSize]
        
        out <- list(population = object@population[sel, , drop = FALSE],
                    fitness = object@fitness[sel])
        return(out)
      }
    
    #####################################
    ##### Parameter Suite ###############
    #####################################
    SelectionMethodByNr <- function(x, ...) {
      if (x == "LinearRank")
        r <- gareal_lrSelection(...)  # linear rank selection
      else if (x == "UnbiasedTournament")
        r <- gareal_tourSelection(k = 4, ...) # unbiased tournament
      else if (x == "FitnessProportional")
        r <- gareal_lsSelection(...) # fitness proportional selection
      else
        stop(paste("Selectionmethod", x, "was not recognized"))
      
      return(r)
    }
    CrossoverMethodByNr <- function(x, object, parents, ...) {
      if (x == "SinglePoint")
        r <-
          gareal_spCrossover(object, parents, ...) # single point crossover
      else if (x == "WholeArithmetic")
        r <-
          gareal_waCrossover(object, parents, ...) # whole arithmetic crossover
      else if (x == "LocalArithmetic")
        r <-
          gareal_laCrossover(object, parents, ...) # local arithmetic crossover
      else if (x == "Blend")
        r <- gareal_blxCrossover(object, parents, ...) # blend crossover
      else if (x == "Component")
        r <- GaussCrossover(object, parents, ...)
      else if (x == "MultiPoint")
        r <- MultiPointCrossover(object, parents, ...)
      else
        stop(paste("Crossovermethod", x, "was not recognized"))
      
      
      if (SortedGauss) {
        # werte auslesen
        
        n = nrow(r$children)
        
        for (i in 1:n) {
          Means = r$children[1, (Modes * 2 + 1):(Modes * 3)]
          sorted = order(Means)
          
          Weights = r$children[i, 1:Modes]
          SDs = r$children[i, (Modes + 1):(Modes * 2)]
          Means = r$children[i, (Modes * 2 + 1):(Modes * 3)]
          sorted = order(Means)
          
          if (any(is.nan(SDs))) {
            print("sd error before sorting in crossover")
          }
          
          r$children[i,] = c(Weights[sorted], SDs[sorted], Means[sorted])
        }
      }
      
      return(r)
    }
    MutationMethodByNr <- function(x, o, p) {
      if (x == "Uniform")
        r <- UniformMutation(o, p) # uniform random mutation
      else if (x == "NonUniform")
        r <- gareal_nraMutation(o, p)# nonuniform random mutation
      else if (x == "Focused")
        r <-
          GaussMutationWithIntensity(o, p) # auch random mutation around solution?
      else if (x == "E+SD_Focused")
        r <- GaussMutationCust(o, p)
      else if (x == "Uniform+Focused")
        r <- TwoPhaseMutation(o, p)
      else
        stop(paste("Mutationmethod", x, "was not recognized"))
      
      if (SortedGauss) {
        # werte auslesen
        Weights = r[1:Modes]
        SDs = r[(Modes + 1):(Modes * 2)]
        Means = r[(Modes * 2 + 1):(Modes * 3)]
        sorted = order(Means)
        
        if (any(is.nan(SDs))) {
          print("sd error before sorting in mutation")
          browser()
        }
        
        return(c(Weights[sorted], SDs[sorted], Means[sorted]))
      }
      
      return(r)
    }
    
    sortedCustomPopulation <- function(object, ...) {
      #r <- matrix(rep(c(Weights, SDs, Means), object@popSize ), byrow=T, ncol=Modes * 3)
      r <- gareal_Population(object, ...)
      
      breaks = tail(head(seq(min(Data), max(Data), length.out = Modes + 2), -1), -1)
      
      for (i in 1:nrow(r)) {
        # erwartungswerte
        k = 1
        for (br in breaks) {
          r[i, (Modes * 2 + k)] = rnorm(1, br, (max(Data) - min(Data)) / (Modes *
                                                                            5))
          
          # standardabweichungen
          r[i, (Modes + k)] = runif(1, 0, (max(Data) - min(Data)) / (Modes *
                                                                       3))
          
          k = k + 1
        }
        
        # überschreiben falls Werte gegeben
        if (!is.null(Weights))
          r[i, 1:Modes] = Weights
        if (!is.null(SDs))
          r[i, (Modes + 1):(Modes * 2)] = SDs
        if (!is.null(Means))
          r[i, (Modes * 2 + 1):(Modes * 3)] = Means
        
      }
      
      #sortieren
      for (i in 1:nrow(r)) {
        Weights = r[i, 1:Modes]
        SDs = r[i, (Modes + 1):(Modes * 2)]
        Means = r[i, (Modes * 2 + 1):(Modes * 3)]
        sorted = order(Means)
        
        r[i, ] = c(Weights[sorted], SDs[sorted], Means[sorted])
      }
      
      
      return(r)
    }
    
    concurrentStartPopulation <- function(object, ...) {
      populations = list()
      for (i in 1:5) {
        V <- DistributionOptimization(Data, Modes, Monitor,)
        
      }
    }
    
    
    concurrentPopulation <- function(object, ...) {
      populations = list()
      fitness = c()
      
      print("Generating Startpopulation...")
      
      for (i in 1:5) {
        # MutationMethod anpassen
        V <- DistributionOptimization(
          Data,
          Modes,
          Monitor = 0,
          SelectionMethod = SelectionMethod,
          MutationMethod = "NonUniform",
          CrossoverMethod = CrossoverMethod,
          PopulationSize = PopulationSize,
          MutationRate = MutationRate,
          Elitism = Elitism,
          CrossoverRate = CrossoverRate,
          Iter = 30,
          OverlapTolerance = OverlapTolerance,
          IsLogDistribution = IsLogDistribution,
          ErrorMethod = ErrorMethod,
          NoBins = NoBins,
          ConcurrentInit = F,
          ParetoRad = ParetoRad
        )
        
        populations[[i]] = V$GA@population
        fitness[i] = V$GA@fitnessValue
        
        print(paste0(round((i / 5) * 100, 2), "% done"))
      }
      bestPopulationIndex = which.max(fitness)
      return(populations[[bestPopulationIndex]])
    }
    
    ##############################
    ##### Bestimmung der Grundkonfiguration
    ##############################
    
    
    
    #if(!PDEMode)
    #  precalc <- Chi2Value_precalc(Data)
    #else
    #  precalc <- PDEError_precalc(Data)
    
    
    if (Monitor == 0)
      ActiveMonitor = NULL
    else
      ActiveMonitor = MonitorFun
    
    ############################
    ##### Start des Trainings
    ###########################
    tryCatch({
      evaluations_count <- 0
      sol <- ga(
        type = "real-valued",
        fitness = function(x) {
          if(evaluations_count == Budget){
            print("Budget reached")
          }
          if(evaluations_count > Budget){
            return(-Inf)
          }
          evaluations_count <<- evaluations_count + 1
          GaussmixtureFitness(x)
        },
        lower = minimums,
        upper = maximums,
        popSize = PopulationSize,
        maxiter = Iter,
        monitor = ActiveMonitor,
        #pmutation = MutationRate, #startMutation, #
        pmutation = function(...) {
          ga_pmutation(..., p0 = MutationRate, p = MutationRate)
        },
        mutation = function(...)
          MutationMethodByNr(MutationMethod, ...),
        crossover = function(object, parents, ...)
          CrossoverMethodByNr(CrossoverMethod, object, parents, ...),
        selection = function(...)
          SelectionMethodByNr(SelectionMethod, ...),
        population = function(...) {
          if (ConcurrentInit)
            concurrentPopulation(...)
          else
            sortedCustomPopulation(...)
        },
        elitism = base::max(1, round(PopulationSize * Elitism)),
        optim = Optim,
        pcrossover = CrossoverRate,
        parallel = F
      )
    },
    interrupt = function(x)
      print("interrupted"),
    finally = {
    })
    
    # extract results
    x = sol@solution[1, ]
    
    Weights = x[1:Modes]
    SDs = x[(Modes + 1):(Modes * 2)]
    Means = x[(Modes * 2 + 1):(Modes * 3)]
    Weights = Weights / sum(Weights)
    
    return(list(
      GA = sol,
      Weights = Weights,
      SDs = SDs,
      Means = Means
    ))
  }
