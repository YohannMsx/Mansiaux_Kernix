set.seed(1234)
##########################
# Question additionnelle #
##########################
# Ces méthodes sont-elles efficaces pour distinguer un excellent vin 
# (score de 7 ou plus) d’un autre vin ? Le seuil est défini de manière arbitraire
wine2 <- wine
wine2$quality <- ifelse(wine$quality>=7,1,0)
# on observe la répartition dans les 2 catégories #
table(wine2$quality)

# division du jeu de données en 2 parties -> train et test
train <- wine2[samples$train,]; test <- wine2[samples$test,]

########################
# partie apprentissage #
########################
# modèle de régression linéaire remplacé par un modèle logistique (glm)
# glm #
.glm <- glm(quality~.,data=train,family=binomial)

# fa #
.fa <- randomForest(factor(quality)~.,data=train, ntree=1000)

# arb #
.arb <- gbm.fit(x=train[,.predictors], y=train[,.outcome], distribution = "bernoulli", 
                n.trees=1000, verbose = F)

# LASSO #
wine_matrix <- data.matrix(train[,c(.predictors,.outcome)])
.lasso <- glmnet(x=wine_matrix[,.predictors],y=wine_matrix[,.outcome],
                 family = "binomial", standardize = T)

###############
# partie test #
###############
# prédiction des donnees tests à partir des modèles construits sur les données train #
.glmtest <- predict(object=.glm, newdata=test,type="link")
.fatest <- predict(object=.fa, newdata=test,type="prob")
.arbtest <- predict(object=.arb, newdata=test, n.trees=1000, type = "link")

test_matrix <- data.matrix(test[,c(.predictors)])
.lassotest <- predict(object=.lasso, newx=test_matrix, type = "link")

###############################
# évaluation des performances #
###############################
# le critère d'évaluation est l'aire sous la courbe ROC (AUC) -> critère à maximiser
# ceci nécessite d'obtenir pour chaque modèle la probabilité modélisée
# que chaque vin soit un "excellent" vin
# le modèle rf renvoie directement cette probabilité
# pour les autres modèles, on doit obtenir cette probabilité à partir
# de la fonction "logit" modélisée
# le code de la fonction Logit2prob est présent dans le script functions.R

.glmprob <- Logit2prob(.glmtest)
.faprob <- .fatest[,2]
.arbprob <- Logit2prob(.arbtest)
.lassoprob <- apply(.lassotest,2,Logit2prob)

# AUC #
.glmauc <- auc(test[,.outcome],.glmprob)
.faauc <- auc(test[,.outcome],.faprob)
.arbauc <- auc(test[,.outcome],.arbprob)
.lassoauc <- apply(.lassoprob,2,function(.col) auc(test[,.outcome],.col))

# on récupère le modèle avec l'AUC la plus élevée : c'est le modèle FA #
.allauc <- c(.glmauc, .faauc, .arbauc, max(.lassoauc)); names(.allauc) <- c(".glmauc", ".faauc", ".arbauc",".lassoauc")
.maxauc <- .allauc[which.max(.allauc)]

# comme dans la première analyse on observe les variables les plus importantes
.fatest <- randomForest(factor(quality)~.,data=test, ntree=1000, importance=T)
.varsImp <- .fatest$importance[,"MeanDecreaseAccuracy"]
.varsImp <- .varsImp[order(.varsImp,decreasing=T)]
barplot(.varsImp,horiz=T,names.arg=F,col=rainbow(11),legend.text=names(.varsImp),xlim=c(0,0.04),xlab="Importance des variables",font=2,cex.lab=1.2)



