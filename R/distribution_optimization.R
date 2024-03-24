library(DataVisualizations)
library(DistributionOptimization)
library(GA)
library(ggplot2)
library(tidyverse)

truck_driving_data <- read_csv("../data/truck_driving_data.csv")$value

chromatogram_time <- read_csv("../data/chromatogram_time.csv")$value

mixture3 <- read_csv("../data/mixture3.csv")$value

textbook_data <- read_csv("../data/textbook_1k.csv")$value

atmosphere_data <- read_csv("../data/atmosphere_data.csv")$value

iris_data <- read_csv("../data/iris_ica.csv")$value

BUDGETS <- c(1000, 2000, 5000, 10000, 15000)

get_fitness <- function(Data, Modes) {
  OverlapTolerance = 0.5
  NoBins = OptimalNoBins(Data)
  Kernels = seq(min(Data), max(Data), length.out = 40)
  breaks = seq(min(Data), max(Data), length.out = NoBins + 1)
  observedBins = hist(Data, breaks = breaks, plot = F)$counts
  GaussmixtureFitness <- function(x) {
    Weights = x[1:Modes]
    SDs = x[(Modes + 1):(Modes * 2)]
    Means = x[(Modes * 2 + 1):(Modes * 3)]
    N = length(Data)
    Weights = Weights / sum(Weights)
    estimatedBins = BinProb4Mixtures(Means, SDs, Weights,
                                     breaks) * N
    norm = estimatedBins
    norm[norm < 1] = 1
    diffssq = (observedBins - estimatedBins) ^ 2
    diffssq[diffssq < 4] = 0
    SimilarityError = sum(diffssq / norm)
    SimilarityError = SimilarityError / N
    a <- OverlapTolerance
    if (OverlapTolerance < 1) {
      OError = OverlapErrorByDensity(Means, SDs, Weights,
                                     Data, Kernels)$OverlapError
    }
    else {
      OError = 0
    }
    if (OError > OverlapTolerance)
      res = 10
    else
      res <- SimilarityError
    return(-res)
  }
  return(GaussmixtureFitness)
}

ga_config <- list(
  "PopulationSize" = 39.21577,
  "MutationRate" = 0.7839187,
  "CrossoverRate" = 0.9000,
  "Elitism" = 0.08389597
)

get_ga_results <- function(Data, Modes, budgets = BUDGETS) {
  budget_to_results = list()
  for (budget in budgets) {
    seeds <- 1:30
    ga_results <- list()
    for (seed in seeds) {
      set.seed(seed)
      Iter = budget %/% 25
      result <-
        DistributionOptimization(
          Data,
          Modes,
          Iter = Iter,
          Budget = budget,
          PopulationSize = ga_config$PopulationSize,
          MutationRate = ga_config$MutationRate,
          CrossoverRate = ga_config$CrossoverRate,
          Elitism = ga_config$Elitism
        )$GA
      ga_results <- c(ga_results, list(result))
    }
    fitness_values <-
      unlist(lapply(ga_results, function(x)
        x@fitnessValue))
    budget_to_results[[as.character(budget)]] <- fitness_values
  }
  return(budget_to_results)
}

get_ga_solutions <- function(Data, Modes) {
  seeds <- c(1)
  budget <- 1000
  ga_results <- list()
  for (seed in seeds) {
    set.seed(seed)
    Iter = budget %/% 50
    result <-
      DistributionOptimization(Data, Modes, Iter = Iter, Budget = budget)$GA
    ga_results <- c(ga_results, list(result))
  }
  solutions <- lapply(ga_results, function(x)
    x@solution)
  return(solutions)
}

read_hms_results <- function(Data, Modes, dataset_name, prefix) {
  fitness <- get_fitness(Data, Modes)
  budget_to_fitness_values <- list()
  for (budget in BUDGETS) {
    file_path <-
      paste(prefix,
            dataset_name,
            "_solutions_",
            as.character(budget),
            ".csv",
            sep = "")
    hms_solutions <- read.csv(file_path, header = FALSE)
    hms_fitness_values <- c()
    for (i in 1:nrow(hms_solutions)) {
      fitness_value <- fitness(unlist(hms_solutions[i , ]))
      hms_fitness_values <- c(hms_fitness_values, c(fitness_value))
    }
    budget_to_fitness_values[[as.character(budget)]] <-
      hms_fitness_values
  }
  return(budget_to_fitness_values)
}


plot_boxplots <-
  function(ga_results, hms_results, dataset_name = "") {
    data_ga <-
      enframe(ga_results, name = "n_eval", value = "scores") %>%
      unnest(scores) %>%
      mutate(algorithm = "DistributionOptimization")

    data_hms <-
      enframe(hms_results, name = "n_eval", value = "scores") %>%
      unnest(scores) %>%
      mutate(algorithm = "HMS")
    data_combined <- bind_rows(data_ga, data_hms)
    ggplot(data_combined, aes(
      x = as.factor(as.integer(n_eval)),
      y = scores * (-1),
      fill = algorithm
    )) +
      geom_boxplot(position = position_dodge(0.8)) +
      xlab("Function Evaluations") +
      ylab("Function Values") +
      # ggtitle(paste("Boxplot of Scores for GA vs HMS", dataset_name)) +
      scale_fill_manual(values = c("DistributionOptimization" = "blue", "HMS" = "red"))
  }

plot_all_boxplots <-
  function(ga_results, hms_results, de_results, cma_es_results, dataset_name = "") {
    data_ga <-
      enframe(ga_results, name = "n_eval", value = "scores") %>%
      unnest(scores) %>%
      mutate(algorithm = "GA")

    data_hms <-
      enframe(hms_results, name = "n_eval", value = "scores") %>%
      unnest(scores) %>%
      mutate(algorithm = "HMS")

    data_de <-
      enframe(de_results, name = "n_eval", value = "scores") %>%
      unnest(scores) %>%
      mutate(algorithm = "iL-SHADE")

    data_cma_es <-
      enframe(cma_es_results, name = "n_eval", value = "scores") %>%
      unnest(scores) %>%
      mutate(algorithm = "BIPOP-CMA-ES")

    data_combined <- bind_rows(data_ga, data_hms, data_de, data_cma_es)
    ggplot(data_combined, aes(
      x = as.factor(as.integer(n_eval)),
      y = scores,
      fill = algorithm
    )) +
      geom_boxplot(position = position_dodge(0.8)) +
      xlab("n_eval") +
      ylab("Scores") +
      ggtitle(paste("Boxplot of Scores for GA vs HMS", dataset_name)) +
      scale_fill_manual(values = c("GA" = "blue", "HMS" = "red", "iL-SHADE" = "green", "BIPOP-CMA-ES" = "purple"))
  }

ga_mixture3_results <- get_ga_results(mixture3, 3)
load("../experiments/results_ga/ga_tuned_mixture3_results.rda")
save(ga_mixture3_results, file = "../experiments/results_ga/ga_tuned_mixture3_results.rda")
ga_truck_driving_results <- get_ga_results(truck_driving_data, 3)
load("../experiments/results_ga/ga_tuned_truck_driving_results.rda")
save(ga_truck_driving_results, file = "../experiments/results_ga/ga_tuned_truck_driving_results.rda")
ga_textbook_results <- get_ga_results(textbook_data, 3)
load("../experiments/results_ga/ga_tuned_textbook_results.rda")
save(ga_textbook_results, file = "../experiments/results_ga/ga_tuned_textbook_results.rda")
ga_iris_results <- get_ga_results(iris_data, 3)
load("../experiments/results_ga/ga_tuned_iris_results.rda")
save(ga_iris_results, file = "../experiments/results_ga/ga_tuned_iris_results.rda")
ga_atmosphere_results <- get_ga_results(atmosphere_data, 5)
load("../experiments/results_ga/ga_tuned_atmosphere_results.rda")
save(ga_atmosphere_results, file = "../experiments/results_ga/ga_tuned_atmosphere_results.rda")
ga_chromatogram_results <- get_ga_results(chromatogram_time, 5)
load("../experiments/results_ga/ga_tuned_chromatogram_results.rda")
save(ga_chromatogram_results, file = "../experiments/results_ga/ga_tuned_chromatogram_results.rda")

PREFIX <- "../experiments/results_hms/"
hms_mixture3_results <-
  read_hms_results(mixture3, 3, "mixture3", PREFIX)
hms_truck_driving_results <-
  read_hms_results(truck_driving_data, 3, "truck_driving_data", PREFIX)
hms_textbook_results <-
  read_hms_results(textbook_data, 3, "textbook_data", PREFIX)
hms_iris_results <-
  read_hms_results(iris_data, 3, "iris_ica", PREFIX)
hms_atmosphere_results <-
  read_hms_results(atmosphere_data, 5, "atmosphere_data", PREFIX)
hms_chromatogram_results <-
  read_hms_results(chromatogram_time, 5, "chromatogram_time", PREFIX)

DE_PREFIX <- "../experiments/results_de_ilshade/"
de_mixture3_results <-
  read_hms_results(mixture3, 3, "mixture3", DE_PREFIX)
de_truck_driving_results <-
  read_hms_results(truck_driving_data, 3, "truck_driving_data", DE_PREFIX)
de_textbook_results <-
  read_hms_results(textbook_data, 3, "textbook_data", DE_PREFIX)
de_iris_results <-
  read_hms_results(iris_data, 3, "iris_ica", DE_PREFIX)
de_atmosphere_results <-
  read_hms_results(atmosphere_data, 5, "atmosphere_data", DE_PREFIX)
de_chromatogram_results <-
  read_hms_results(chromatogram_time, 5, "chromatogram_time", DE_PREFIX)

CMA_ES_PREFIX <- "../experiments/results_cma_es_bipop/"
cma_es_mixture3_results <-
  read_hms_results(mixture3, 3, "mixture3", CMA_ES_PREFIX)
cma_es_truck_driving_results <-
  read_hms_results(truck_driving_data, 3, "truck_driving_data", CMA_ES_PREFIX)
cma_es_textbook_results <-
  read_hms_results(textbook_data, 3, "textbook_data", CMA_ES_PREFIX)
cma_es_iris_results <-
  read_hms_results(iris_data, 3, "iris_ica", CMA_ES_PREFIX)
cma_es_atmosphere_results <-
  read_hms_results(atmosphere_data, 5, "atmosphere_data", CMA_ES_PREFIX)
cma_es_chromatogram_results <-
  read_hms_results(chromatogram_time, 5, "chromatogram_time", CMA_ES_PREFIX)

hms_results = list(
  "Mixture 3" = hms_mixture3_results,
  "Truck Driving" = hms_truck_driving_results,
  "Textbook" = hms_textbook_results,
  "Iris" = hms_iris_results,
  "Atmosphere" = hms_atmosphere_results,
  "Chromatogram" = hms_chromatogram_results
)

ga_results = list(
  "Mixture 3" = ga_mixture3_results,
  "Truck Driving" = ga_truck_driving_results,
  "Textbook" = ga_textbook_results,
  "Iris" = ga_iris_results,
  "Atmosphere" = ga_atmosphere_results,
  "Chromatogram" = ga_chromatogram_results
)

de_results = list(
  "Mixture 3" = de_mixture3_results,
  "Truck Driving" = de_truck_driving_results,
  "Textbook" = de_textbook_results,
  "Iris" = de_iris_results,
  "Atmosphere" = de_atmosphere_results,
  "Chromatogram" = de_chromatogram_results
)

cma_es_results = list(
  "Mixture 3" = cma_es_mixture3_results,
  "Truck Driving" = cma_es_truck_driving_results,
  "Textbook" = cma_es_textbook_results,
  "Iris" = cma_es_iris_results,
  "Atmosphere" = cma_es_atmosphere_results,
  "Chromatogram" = cma_es_chromatogram_results
)

budget_comparison <- function(budget) {
  rows <- list()
  for (dataset_name in names(ga_results)) {
    ga_results_dataset <-
      ga_results[[dataset_name]][[as.character(budget)]]
    hms_results_dataset <-
      hms_results[[dataset_name]][[as.character(budget)]]
    de_results_dataset <- de_results[[dataset_name]][[as.character(budget)]]
    cma_es_results_dataset <- cma_es_results[[dataset_name]][[as.character(budget)]]
    row <-
      list(
        "hms_mean" = mean(hms_results_dataset),
        "ga_mean" = mean(ga_results_dataset),
        "de_mean" = mean(de_results_dataset),
        "cma_es_mean" = mean(cma_es_results_dataset),
        "hms_median" = median(hms_results_dataset),
        "ga_median" = median(ga_results_dataset),
        "de_median" = median(de_results_dataset),
        "cma_es_median" = median(cma_es_results_dataset),
        "hms_sd" = sd(hms_results_dataset),
        "ga_sd" = sd(ga_results_dataset),
        "de_sd" = sd(de_results_dataset),
        "cma_es_sd" = sd(cma_es_results_dataset),
        "dataset" = dataset_name,
        "budget" = budget
      )
    rows <- c(rows, list(row))
  }
  df <- do.call(rbind, lapply(rows, function(x) as.data.frame(t(x), stringsAsFactors = FALSE)))
  return(df)
}


plot_all_boxplots(
  ga_mixture3_results,
  hms_mixture3_results,
  de_mixture3_results,
  cma_es_mixture3_results,
  "Mixture3"
)
plot_all_boxplots(
  ga_truck_driving_results,
  hms_truck_driving_results,
  de_truck_driving_results,
  cma_es_truck_driving_results,
  "Truck Driving"
)
plot_all_boxplots(
  ga_textbook_results,
  hms_textbook_results,
  de_textbook_results,
  cma_es_textbook_results,
  "Textbook"
)
plot_all_boxplots(ga_iris_results,
                  hms_iris_results,
                  de_iris_results,
                  cma_es_iris_results,
                  "Iris ICA")
plot_all_boxplots(
  ga_atmosphere_results,
  hms_atmosphere_results,
  de_atmosphere_results,
  cma_es_atmosphere_results,
  "Atmosphere"
)
plot_all_boxplots(
  ga_chromatogram_results,
  hms_chromatogram_results,
  de_chromatogram_results,
  cma_es_chromatogram_results,
  "Chromatogram"
)

plot_boxplots(ga_mixture3_results, hms_mixture3_results, "Mixture3")
plot_boxplots(ga_truck_driving_results,
              hms_truck_driving_results,
              "Truck Driving")
plot_boxplots(ga_textbook_results, hms_textbook_results, "Textbook")
plot_boxplots(ga_iris_results, hms_iris_results, "Iris ICA")
plot_boxplots(ga_atmosphere_results, hms_atmosphere_results, "Atmosphere")
plot_boxplots(ga_chromatogram_results,
              hms_chromatogram_results,
              "Chromatogram")
