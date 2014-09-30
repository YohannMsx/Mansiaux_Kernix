# URL du jeu de données : http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# Jeu de données relatif à différents vins rouges
# Variables indépendantes : 11 propriétés physicochimiques 
# Variable dépendante : Score de qualité (0-10)

rm(list=ls(all=T))
set.seed(1234)

### librairies nécessaires + fonctions ###
library(randomForest); library(gbm); library(glmnet); library(RColorBrewer); library(pROC)
# peut être nécessaire de préciser le répertoire de travail : setwd()
source("./functions.R")

### chargement des donnees ###
wine <- read.csv("./winebase/winequality-red.csv",h=T,sep=";")
.outcome <- "quality"; .predictors <- colnames(wine)[!colnames(wine) %in% .outcome]

# Distribution du score de qualité 
summary(wine$quality)
palette_bleus <- brewer.pal(n=6,name="Blues")
barplot(table(wine$quality),main="",ylab="Fréquence",xlab="Qualité",font=2,cex.lab=1.2, col =palette_bleus)

# Analyse en 2 étapes :
# 1) Différents algorithmes sont testés : 
# - 2 méthodes de data mining : Forêt aléatoire (fa) et Arbres de régression boostés (arb) 
# -> intéret de ces méthodes : fournissent des scores d'importance permettant d'identifier les prédicteurs les plus importants
# - 1 méthode de régression linéaire (rl)
# - 1 méthode de régression pénalisée : LASSO (lasso)
# Objectif numéro 1 : obtenir le meilleur modèle prédictif possible parmi les 4 algorithmes
# Pour cela le jeu de données est divisé en 2 parties : 1 partie apprentissage (Train) pour construire les modèles
# 1 partie test (Test) pour évaluer leurs performances prédictives
# 2) Une fois le meilleur algorithme identifié, on cherche à savoir quelles sont les variables
# les plus importantes pour prédire le score de qualité

###########
# Etape 1 #
###########

# division du jeu de données en 2 parties -> train et test (code de la fonction dans le script functions.R)
samples <- TrainTestSamples(data=wine,propTrain=0.7)
train <- wine[samples$train,]; test <- wine[samples$test,]

########################
# partie apprentissage #
########################
# rl #
.rl <- lm(quality~.,data=train)

# fa #
.fa <- randomForest(quality~.,data=train, ntree=1000)

# arb #
.arb <- gbm.fit(x=train[,.predictors], y=train[,.outcome], distribution = "gaussian", 
                n.trees=1000, verbose = F)

# LASSO : nécessite que les données soient sous forme "matrix"
wine_matrix <- data.matrix(train[,c(.predictors,.outcome)])
# les prédicteurs doivent être standardisés
.lasso <- glmnet(x=wine_matrix[,.predictors],y=wine_matrix[,.outcome],
                 family = "gaussian", standardize = T)

###############
# partie test #
###############
# prédiction des données test à partir des modèles construits sur les données train #
.rltest <- predict(object=.rl, newdata=test)
.fatest <- predict(object=.fa, newdata=test)
.arbtest <- predict(object=.arb, newdata=test, n.trees=1000)

test_matrix <- data.matrix(test[,c(.predictors)])
.lassotest <- predict(object=.lasso, newx=test_matrix)

###############################
# évaluation des performances #
###############################
# (code de la fonction dans le script functions.R)
# pour chaque méthode on calcule le MSR (mean square residuals) -> 
# moyenne des résidus (écart entre prédictions et observations) au carré

.rlmsr <- MSR(predicted=.rltest, observed=test[,.outcome])
.famsr <- MSR(predicted=.fatest, observed=test[,.outcome])
.arbmsr <- MSR(predicted=.arbtest, observed=test[,.outcome])

# le lasso a la particularité de tester simultanément un ensemble de paramètres de pénalisation
# x paramètres de pénalisation <-> x jeux de coefficients estimés <-> x calculs de MSR
.lassomsr <- apply(.lassotest, 2, function(.col) MSR(predicted=.col, observed=test[,.outcome]))

# on récupère le modèle avec le MSR le plus faible : c'est le modèle FA #
.allmsr <- c(.rlmsr, .famsr, .arbmsr, min(.lassomsr)); names(.allmsr) <- c(".rlmsr", ".famsr", ".arbmsr",".lassomsr")
.minmsr <- .allmsr[which.min(.allmsr)]

###########
# Etape 2 #
###########
# l'instruction importance = T permet de calculer l'importance de chacune des variables 
# dans la prédiction du score de qualité
.fatest <- randomForest(quality~.,data=test, ntree=1000, importance=T)
.varsImp <- .fatest$importance[,"%IncMSE"]
.varsImp <- .varsImp[order(.varsImp,decreasing=T)]

barplot(.varsImp,horiz=T,names.arg=F,col=rainbow(11),legend.text=names(.varsImp),
        xlim=c(0,0.2),xlab="Importance des variables",font=2,cex.lab=1.2)

# on observe la nature de la relation entre les 3 variables les plus importantes 
# et la variable quality 
# on trace également la droite issue d'un modèle de régression linéaire étudiant 
# la relation entre la variable quality et la variable prédictive étudiée pour avoir
# une idée approximative du sens de la relation
plot(wine$alcohol,wine$quality,ty="p",cex=.5, xlab="Alcool", ylab="Qualité", cex.lab=1.1, font.lab=2)
abline(lm(quality~alcohol,data=wine),col="red",lwd=2)

plot(wine$sulphates,wine$quality,ty="p",cex=.5, xlab="Sulfates", ylab="Qualité", cex.lab=1.1, font.lab=2)
abline(lm(quality~sulphates,data=wine),col="red",lwd=2)

plot(wine$volatile.acidity,wine$quality,ty="p",cex=.5,xlab="Acidité volatile", ylab="Qualité", cex.lab=1.1, font.lab=2)
abline(lm(quality~volatile.acidity,data=wine),col="red",lwd=2)

##########################
# Question additionnelle #
##########################
# Ces méthodes sont-elles efficaces pour distinguer un excellent vin 
# (score de 7 ou plus) d’un autre vin ?
# -> voir script analyse_additionnelle.R
