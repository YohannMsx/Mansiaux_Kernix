TrainTestSamples <- function(data, propTrain, seed=1234) {
  # renvoie 2 échantillons : 1 échantillon d'apprentissage "train" et un échantillon test
  # propTrain permet de fixer la proportion d'individus inclus dans le jeu 'train'
  set.seed(seed)
  indiv <- 1:nrow(data)
  train <- sample(indiv,size=propTrain*length(indiv),F)
  test <- indiv[!indiv %in% train]
  return(list("train"=train,"test"=test))
}

MSR <- function(predicted,observed) {
  # MSR : mean square residuals
  # observed correspond aux valeurs observées, predicted correspond aux valeurs prédites
  msr <- sum((observed - predicted)^2)/length(predicted)
  return(msr)
}

Logit2prob <- function(logit) return (exp(logit)/(1+exp(logit)))
# à partir d'une fonction logit calculée par un modèle, renvoie la probabilité 
# de présence de la variable réponse correspondante