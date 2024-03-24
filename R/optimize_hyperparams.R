library(DataVisualizations)
library(DistributionOptimization)
library(GA)
library(ggplot2)
library(tidyverse)
library(rBayesianOptimization)

chromatogram_time <- read_csv("../data/chromatogram_time.csv")$value

search_bounds <- list(
  PopulationSize = c(10, 250),
  MutationRate = c(0.1, 0.9),
  CrossoverRate = c(0.1, 0.9),
  Elitism = c(0.01, 0.2)
)

fitness <- function(
    PopulationSize,
    MutationRate,
    CrossoverRate,
    Elitism
  ) {
  Data <- chromatogram_time
  Modes <- 5
  seeds <- 1:10
  budget <- 5000
  ga_results <- list()
  for (seed in seeds) {
    set.seed(seed)
    Iter = budget %/% PopulationSize + 10
    result <-
      DistributionOptimization(
        Data,
        Modes,
        Iter = Iter,
        Budget = budget,
        PopulationSize = PopulationSize,
        MutationRate = MutationRate,
        CrossoverRate = CrossoverRate,
        Elitism = Elitism
        )$GA
    ga_results <- c(ga_results, list(result))
  }
  fitness_values <- unlist(lapply(ga_results, function(x) x@fitnessValue))
  result <- list(Score = mean(fitness_values), Pred = 0)
  return(result)
}

bayes_optimization <-
  BayesianOptimization(
    FUN = fitness,
    bounds = search_bounds,
    init_points = 2,
    n_iter = 100,
    acq = "ucb"
  )

# Best Parameters Found:
# Round = 86	PopulationSize = 39.21577	MutationRate = 0.7839187	CrossoverRate = 0.9000	Elitism = 0.08389597	Value = -0.0001112494
