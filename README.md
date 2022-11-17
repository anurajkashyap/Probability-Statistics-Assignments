---
title: "**Probability and Statistics with R**"
author: "Anuraj Kashyap, Akshay Thorave, Praveen Kumar"
date: "Nov 16-2022"
output: pdf_document
---

## Problem 1

Suppose $X$ denote the number of goals scored by home team in premier league. We can assume $X$ is a random variable. Then we have to build the probability distribution to model the probability of number of goals. Since $X$ takes value in $\mathbb{N}=\{0,1,2,\cdots\}$, we can consider the geometric progression sequence as possible candidate model, i.e.,
$$
S=\{a,ar,ar^2,ar^{3},\cdots\}.
$$
But we have to be careful and put proper conditions in place and modify $S$ in such a way so that it becomes proper probability distributions. 

#### 1. Figure out the necesary conditions and define the probability distribution model using $S$.

#### Answer: 

In order to define the probability distribution model using $S$, following necessary conditions must be       satisfied.
\begin{enumerate}
\item $0 \leq P(X = i) \leq 1, \forall~ i$.
\item $\Sigma_{i = 0}^{\infty} P(X = i) = 1$.
\end{enumerate}

According to the given model $P(X = i) = ar^{i}$.

Therefore,
$$0 \leq a \leq 1 ~~~ \& ~~~ 0\leq r < 1$$
Now,
$$
\Sigma_{i = 0}^{\infty} P(X = i) = 1 $$ 
$$\implies \Sigma_{i = 0}^{\infty} ar^{i} = 1$$
$$\implies r = 1 - a$$
Therefore,
$P(X = i) = a(1-a)^{i}$.


#### 2.Check if mean and variance exists for the probability model.
#### Answer:

$E(X) = \Sigma_{i = 0}^{\infty} iP(X = i) =\Sigma_{i = 0}^{\infty} a(1-a)^{i} = \frac{1-a}{a}$.

Therefore, mean = $E(x) = \frac{1-a}{a}$.

variance = $Var(X) = \Sigma_{i = 0}^{\infty} (i-E(X))^{i} = \frac{1-a}{a^{2}}$.

Therefore, Variance = $\frac{1-a}{a^{2}}$.

Hence, mean and variance exists for the probability model.


#### 3. Can you find the analytically expression of mean and variance.

#### Answer: 
 
 Yes, we can analytically find the expression of mean and variance as follows.
 
 mean = $E(x) = \frac{1-a}{a}$.
 
 Variance = $\frac{1-a}{a^{2}}$.
 
### 4. 
From historical data we found the following summary statistics

\begin{table}[ht]
\centering
     \begin{tabular}{c|c|c|c}\hline
     mean &  median & variance & total number of matches\\ \hline
     1.5 & 1 & 2.25 & 380\\ \hline
     \end{tabular}
\end{table}

Using the summary statistics and your newly defined probability distribution model find the following:

#### a. What is the probability that home team will score at least one goal?
#### b. What is the probability that home team will score at least one goal but less than four goal?

#### Answer:

According to given historical data,
     
$$mean = 1.5$$ 
$$\implies \frac{1-a}{a} = 1.5$$
$$ \implies a = 0.4 $$
Therefore, $X \sim geom(0.4)$.

#### a.

probability that home team will score atleast one goal,

 $= p(X\geq 1) = 1 - p(X = 0)$
```{r}
p = 1 - dgeom(0,0.4)
print(p)
```
#### b.

probability that home team will score at least one goal but less than four goal,

$= p(1 \leq X < 4) =  p(X = 1) + p(X = 2) + p(X = 3)$

```{r}
p = dgeom(1,0.4) + dgeom(2,0.4) + dgeom(3,0.4)
print(p)
```

#### 5. Suppose on another thought you want to model it with off-the shelf Poisson probability models. Under the assumption that underlying distribution is Poisson probability find the above probabilities, i.e.,
#### a. What is the probability that home team will score at least one goal?
#### b. What is the probability that home team will score at least one goal but less than four goal?

#### Answer:

Now, we want to model above distribution with poisson probability model. Assume that, the underlying distribution is poisson probability distribution.

i.e Assume that, X~poisson($\lambda$),   where $\lambda$ = average number of goals = mean.

According to summary statistics, mean = 1.5.

$\implies \lambda = 1.5$.

Thus, 
$X \sim poisson(1.5)$

#### a.
probability that home team will score at least one goal,

$= p(X\geq 1) = 1 - p(X = 0)$

```{r}
p = 1 - dpois(0,1.5)
print(p)
```
#### b.
probability that home team will score at least one goal but less than four goal,

$= p(1 \leq X < 4) =  p(X = 1) + p(X = 2) + p(X = 3)$

```{r}
p = dpois(1,1.5) + dpois(2,1.5) + dpois(3,1.5)
print(p)
```


#### 6. Which probability model you would prefer over another?

#### Answer:

I prefer the poisson model over the newly defined probability model for this given data.












#### 7. Write down the likelihood functions of your newly defined probability models and Poisson models. Clearly mention all the assumptions that you are making.

#### Answer:

i) Likelihood function for newly defined probability model.

Let $x_{1}, x_{2}, x_{3}, \cdots , x_{n}$ be a observed sample.

Then the likelihood function is given by,
\begin{align}
f(x_{1}, x_{2}, x_{3}, \cdots , x_{n} | \hat{a}) &= f(x_{1}| \hat{a})f(x_{2}| \hat{a})f(x_{3}| \hat{a}) \cdots f(x_{n}| \hat{a}) \\
&= \hat{a}(1-\hat{a})^{x_{1}}\hat{a}(1-\hat{a})^{x_{2}}\hat{a}(1-\hat{a})^{x_{3}}\cdots\hat{a}(1-\hat{a})^{x_{n}} \\ 
&= \hat{a}^{n}(1-\hat{a})^{\Sigma_{i=1}^{\infty}x_{i}}
\end{align}

ii) Likelihood function for poisson model.

Let $x_{1}, x_{2}, x_{3}, \cdots , x_{n}$ be a observed sample.

Then the likelihood function is given by,
\begin{align}
f(x_{1}, x_{2}, x_{3}, \cdots , x_{n} | \hat{\lambda}) &= f(x_{1}| \hat{\lambda})f(x_{2}| \hat{\lambda})f(x_{3}| \hat{\lambda}) \cdots f(x_{n}| \hat{\lambda})\\
&= \frac{e^{- \hat{\lambda}} \hat{\lambda}^{x_{1}}}{x_{1}!}\frac{e^{- \hat{\lambda}} \hat{\lambda}^{x_{2}}}{x_{2}!}\frac{e^{- \hat{\lambda}} \hat{\lambda}^{x_{3}}}{x_{3}!}\cdots\frac{e^{- \hat{\lambda}} \hat{\lambda}^{x_{n}}}{x_{n}!}\\
&= \frac{e^{-n\hat{\lambda}} \hat{\lambda}^{\Sigma_{i=1}^{n}x_{i}}}{\Pi_{i = 1}^{n} x_{i}!}
\end{align}


#### Assumptions:

\begin{itemize}
\item Number of goals scored in a match is independent of the number of goals scored in every other match.
\item Number of goals in every match is identically distributed.
\item Assume that no other factors is affecting the number of goals scored in a match.
\end{itemize}

## Problem 2 : Simulation Study to Understand Sampling Distribution

**Part A**
Suppose $X_1,X_2,\cdots,X_n\stackrel{iid}{\sim} Gamma(\alpha,\sigma)$, with pdf as
$$
f(x | \alpha,\sigma)=\frac{1}{\sigma^{\alpha}\Gamma(\alpha)}e^{- x/\sigma}x^{\alpha-1},~~~~0<x<\infty,
$$
The mean and variance are $E(X)=\alpha\sigma$ and $Var(X)=\alpha\sigma^2$. Note that `shape = ` $\alpha$ and `scale = ` $\sigma$.

1. Write a `function` in `R` which will compute the MLE of $\theta=\log(\alpha)$ using `optim` function in `R`. You can name it `MyMLE`

```{r setup, include=TRUE}
MyMLE=function(n,shape,scale)
{
  n<<-n
  
  Negloglike=function(data,theta)
  {
    alpha = exp(theta[1])
    sigma = exp(theta[2])
    
    l=0
    for(i in 1:n)
    {
      l=l+log(dgamma(data[i], shape = alpha, scale = sigma))
      
    }
    return(-l)
  }
  
  theta=c(0.1,0.1)
  
  
  sim=rgamma(n,shape,scale)
  data=sim
  fit = optim(par=theta,Negloglike,data=sim)
  
  alpha_hat = exp(fit$par[1])
  
  return(log(alpha_hat))
}
```



2. Choose `n=20`, and `alpha=1.5` and `sigma=2.2`
     (i) Simulate $\{X_1,X_2,\cdots,X_n\}$ from `rgamma(n=20,shape=1.5,scale=2.2)`
    
```{r include=TRUE}
rgamma(20,1.5,scale=2.2)
```

     (ii) Apply the `MyMLE` to estimate $\theta$ and append the value in a vector
```{r warning=FALSE, include=TRUE}
x=MyMLE(20,1.5,2.2)
x
```     
     
     
     (iii) Repeat the step (i) and (ii) 1000 times
```{r warning=FALSE, include=TRUE}
for (i in 1:1000){
  
  x=append(x,MyMLE(20,1.5,2.2))
}
```     
     (iv) Draw histogram of the estimated MLEs of $\theta$.
```{r warning=FALSE, include=TRUE}
hist(x)
```     
     (v) Draw a vertical line using `abline` function at the true value of $\theta$.
```{r warning=FALSE, include=TRUE}
hist(x)
abline(v=log(1.5),col="blue")
```

     (vi) Use `quantile` function on estimated $\theta$'s to find the 2.5 and 97.5-percentile points. 
```{r warning=FALSE, include=TRUE}
y=quantile(x, probs = c(.025, .975))
y
```
3.  Choose `n=40`, and `alpha=1.5` and repeat the (2).
##
     (i) Simulate $\{X_1,X_2,\cdots,X_n\}$ from `rgamma(n=40,shape=1.5,scale=2.2)`
    
```{r include=TRUE}
rgamma(40,1.5,scale=2.2)
```

     (ii) Apply the `MyMLE` to estimate $\theta$ and append the value in a vector
```{r warning=FALSE, include=TRUE}
x=MyMLE(40,1.5,2.2)
x
```     
     
     
     (iii) Repeat the step (i) and (ii) 1000 times
```{r warning=FALSE, include=TRUE}
for (i in 1:1000){
  
  x=append(x,MyMLE(40,1.5,2.2))
}
```     
     (iv) Draw histogram of the estimated MLEs of $\theta$.
```{r warning=FALSE, include=TRUE}
hist(x)
```     
     (v) Draw a vertical line using `abline` function at the true value of $\theta$.
```{r warning=FALSE, include=TRUE}
hist(x)
abline(v=log(1.5),col="blue")
```

     (vi) Use `quantile` function on estimated $\theta$'s to find the 2.5 and 97.5-percentile points. 
```{r warning=FALSE, include=TRUE}
y=quantile(x, probs = c(.025, .975))
y
```
##
4.  Choose `n=100`, and `alpha=1.5` and repeat the (2).
     (i) Simulate $\{X_1,X_2,\cdots,X_n\}$ from `rgamma(n=100,shape=1.5,scale=2.2)`
    
```{r include=TRUE}
rgamma(100,1.5,scale=2.2)
```

     (ii) Apply the `MyMLE` to estimate $\theta$ and append the value in a vector
```{r warning=FALSE, include=TRUE}
x=MyMLE(100,1.5,2.2)
x
```     
     
     
     (iii) Repeat the step (i) and (ii) 1000 times
```{r warning=FALSE, include=TRUE}
for (i in 1:1000){
  
  x=append(x,MyMLE(100,1.5,2.2))
}
```     
     (iv) Draw histogram of the estimated MLEs of $\theta$.
```{r warning=FALSE, include=TRUE}
hist(x)
```     
     (v) Draw a vertical line using `abline` function at the true value of $\theta$.
```{r warning=FALSE, include=TRUE}
hist(x)
abline(v=log(1.5),col="blue")
```

     (vi) Use `quantile` function on estimated $\theta$'s to find the 2.5 and 97.5-percentile points. 
```{r warning=FALSE, include=TRUE}
y=quantile(x, probs = c(.025, .975))
y
```

5. Check if the gap between 2.5 and 97.5-percentile points are shrinking as sample size `n` is increasing?
```{r warning=FALSE, include=TRUE}
#Yes, It does.
```

*Hint*: Perhaps you should think of writing a single `function` where you will provide the values of `n`, `sim_size`, `alpha` and `sigma`; and it will return the desired output. 

## Problem 3: Analysis of `faithful` datasets.


Consider the `faithful` datasets:
```{r}
attach(faithful)
hist(faithful$waiting,xlab = 'waiting',probability = T,col='pink',main='')
```

Fit following three models using MLE method and calculate **Akaike information criterion** (aka., AIC) for each fitted model. Based on AIC decides which model is the best model? Based on the best model calculate the following probability
$$
\mathbb{P}(60<\texttt{waiting}<70)
$$
```{r}
data = sort(faithful$waiting)
```

# (i) **Model 1**:
$$
f(x)=p*Gamma(x|\alpha,\sigma_1)+(1-p)N(x|\mu,\sigma_2^2),~~0<p<1
$$

```{r}

loglike1 = function(theta,data){
   alpha = exp(theta[1])
   beta = exp(theta[2])
   mu = theta[3]
   sigma = exp(theta[4])
   p = exp(theta[5])/(1+exp(theta[5]))
   n = length(data)
   l=0
   for(i in 1:n){
      l = l + log(p*dgamma(data[i],shape = alpha, rate = beta)
                  +(1-p)*dnorm(data[i], mean = mu, sd = sigma))
   }
   return(-l)
}

theta_initial=c(4.4,0.47,75,8,0.35)

fit = optim(theta_initial, loglike1, data = data, control = list(maxit=2000))

theta_hat = fit$par
alpha_hat = exp(theta_hat[1])
beta_hat = exp(theta_hat[2])
mu_hat = theta_hat[3]
sigma_hat = exp(theta_hat[4])
p_hat = exp(theta_hat[5])/(1+exp(theta_hat[5]))

d_mle = p_hat*dgamma(data, shape = alpha_hat, rate = beta_hat)+
   (1-p_hat)*dnorm(data, mean = mu_hat,sd = sigma_hat)

hist(data, probability = T, ylim = c(0, 0.05))
lines(data, d_mle,lwd=3,col='green')

AIC = 2*length(fit$par) - 2*(-fit$value)
## AIC Value for model 1
AIC
```

## AIC value for model 1 is 2076.506




# (ii) **Model 2**:
$$
f(x)=p*Gamma(x|\alpha_1,\sigma_1)+(1-p)Gamma(x|\alpha_2,\sigma_2),~~0<p<1
$$
```{r}
loglike2 = function(theta,data){
   alpha1 = exp(theta[1])
   beta1 = exp(theta[2])
   alpha2 = exp(theta[3])
   beta2 = exp(theta[4])
   p = exp(theta[5])/(1+exp(theta[5]))
   n = length(data)
   l=0
   for(i in 1:n){
      l = l + log(p*dgamma(data[i],shape = alpha1, rate = beta1)
                  +(1-p)*dgamma(data[i], shape = alpha2, rate = beta2))
   }
   return(-l)
}

theta_initial = c(4,0,4.4,0,0.35)

fit = optim(theta_initial, loglike2, data = data, control = list(maxit=2000))

theta_hat = fit$par
alpha1_hat = exp(theta_hat[1])
beta1_hat = exp(theta_hat[2])
alpha2_hat = exp(theta_hat[3])
beta2_hat = exp(theta_hat[4])
p_hat = exp(theta_hat[5])/(1+exp(theta_hat[5]))

d_mle = p_hat*dgamma(data, shape = alpha1_hat, rate = beta1_hat)+
   (1-p_hat)*dgamma(data, shape = alpha2_hat, rate = beta2_hat)

hist(data, probability = T, ylim = c(0, 0.05))
lines(data, d_mle,lwd=3,col='red')

AIC = 2*length(fit$par) - 2*(-fit$value)
## AIC Value for model 1
AIC
```


## AIC value for model 2 is 2076.117



# (iii) **Model 3**:
$$
f(x)=p*logNormal(x|\mu_1,\sigma_1^2)+(1-p)logNormal(x|\mu_1,\sigma_1^2),~~0<p<1
$$

```{r}
loglike3 = function(theta,data){
   mu1 = theta[1]
   sigma1 = exp(theta[2])
   mu2 = theta[3]
   sigma2 = exp(theta[4])
   p = exp(theta[5])/(1+exp(theta[5]))
   n = length(data)
   l=0
   for(i in 1:n){
      l = l + log(p*dlnorm(data[i],meanlog = mu1,sdlog = sigma1)
             +(1-p)*dlnorm(data[i],meanlog = mu2,sdlog = sigma2))
   }
   return(-l)
}

theta_initial = c(2.76,-2.25,4.4,-2.6,0.35)

fit = optim(theta_initial, loglike3, data = data, control = list(maxit=2000))

theta_hat = fit$par
mu1_hat = theta_hat[1]
sigma1_hat = exp(theta_hat[2])
mu2_hat = theta_hat[3]
sigma2_hat = exp(theta_hat[4])
p_hat = exp(theta_hat[5])/(1+exp(theta_hat[5]))

d_mle = p_hat*dlnorm(data,meanlog = mu1_hat,sdlog = sigma1_hat)+
      (1-p_hat)*dlnorm(data,meanlog = mu2_hat,sdlog = sigma2_hat)

hist(data, probability = T, ylim = c(0, 0.05))
lines(data, d_mle,lwd=3,col='blue')

AIC = 2*length(fit$par) - 2*(-fit$value)
## AIC Value for model 1
AIC


```
## AIC value for model 3 is 2075.433



# Akaike information criterion(AIC)

Suppose that we have a statistical model of some data. Let k be the number of estimated parameters in the model. Let ${\displaystyle {\hat {L}}}$ be the maximized value of the likelihood function for the model. Then the AIC value of the model is the following.

$${\displaystyle \mathrm {AIC} \,=\,2k-2\ln({\hat {L}})}$$
Given a set of candidate models for the data, the preferred model is the one with the minimum AIC value. 

## Conclusion: 

Comparing AIC values for the three given models, we can observe that AIC value of model 3 is minimum among them making it the best model for the given data. 


# Required probability using best model.

```{r}
dMix = function(x,theta){
  mu1 = theta[1]
  sigma1 = theta[2]
  mu2 = theta[3]
  sigma2 = theta[4]
  p = theta[5]
  f = p*dlnorm(x,meanlog = mu1,sdlog = sigma1)+(1-p)*dlnorm(x, meanlog = mu2, sdlog = sigma2)
  return(f)
}

integrate(dMix,60,70,c(mu1_hat,sigma1_hat,mu2_hat,sigma2_hat,p_hat))

```

$$
\mathbb{P}(60<\texttt{waiting}<70) = 0.09112692
$$



## Problem 4: Modelling Insurance Claims

Consider the `Insurance` datasets in the `MASS` package. The data given in data frame `Insurance` consist of the numbers of policyholders of an insurance company who were exposed to risk, and the numbers of car insurance claims made by those policyholders in the third quarter of 1973.

This data frame contains the following columns:

`District` (factor): district of residence of policyholder (1 to 4): 4 is major cities.

`Group` (an ordered factor): group of car with levels <1 litre, 1–1.5 litre, 1.5–2 litre, >2 litre.

`Age` (an ordered factor): the age of the insured in 4 groups labelled <25, 25–29, 30–35, >35.

`Holders` : numbers of policyholders.

`Claims` : numbers of claims

```{r}
library(MASS)
plot(Insurance$Holders,Insurance$Claims
     ,xlab = 'Holders',ylab='Claims',pch=20)
grid()
```

**Note**: If you use built-in function like `lm` or any packages then no points will be awarded.

**Part A**: We want to predict the `Claims` as function of `Holders`. So we want to fit the following models:
$$
\texttt{Claims}_i=\beta_0 + \beta_1~\texttt{Holders}_i + \varepsilon_i,~~~i=1,2,\cdots,n
$$
*Assume* : $\varepsilon_i\sim N(0,\sigma^2)$. Note that $\beta_0,\beta_1 \in\mathbb{R}$ and $\sigma \in \mathbb{R}^{+}$.

The above model can alse be re-expressed as,
$$
\texttt{Claims}_i\sim N(\mu_i,\sigma^2),~~where
$$
$$
\mu_i =\beta_0 + \beta_1~\texttt{Holders}_i + \varepsilon_i,~~~i=1,2,\cdots,n
$$


(i) Clearly write down the negative-log-likelihood function in `R`. Then use `optim` function to estimate MLE of $\theta=(\beta_0,\beta_1,\sigma)$



```{r warning=FALSE, include=TRUE}
library(SciViews)
library(MASS)
```

```{r warning=FALSE, include=TRUE}
library(jmuOutlier)
Holders=Insurance$Holders
Claims=Insurance$Claims
data=data.frame(cbind(Claims,Holders))
data=data[-61,]
n=length(Holders)-1
y=data[,1]
x=data[,2]
```

```{r warning=FALSE, include=TRUE}
Negloglike=function(data,theta)
{
  l=0
  for(i in 1:n)
  {
    l=l+log(dnorm(y[i], theta[1]+theta[2]*x[i],theta[3]))
    
  }
  return(-l)
}
theta=c(0.1,0.1,50)
fit=optim(theta,Negloglike,data=data)
##Estimated value of theta is:
c(fit$par[1],fit$par[2],fit$par[3])
```


(ii) Calculate **Bayesian Information Criterion** (BIC) for the model.


```{r warning=FALSE, include=TRUE}
BIC_A=ln(n)*(length(fit$par))+2*fit$value
#BIC value is:
BIC_A
```

**Part B**: Now we want to fit the same model with change in distribution:
$$
\texttt{Claims}_i=\beta_0 + \beta_1~\texttt{Holders}_i + \varepsilon_i,~~~i=1,2,\cdots,n
$$
  Assume : $\varepsilon_i\sim Laplace(0,\sigma^2)$. Note that $\beta_0,\beta_1 \in\mathbb{R}$ and $\sigma \in \mathbb{R}^{+}$.

(i) Clearly write down the negative-log-likelihood function in `R`. Then use `optim` function to estimate MLE of $\theta=(\beta_0,\beta_1,\sigma)$



```{r warning=FALSE, include=TRUE}
Negloglike=function(data,theta)
{
  l=0
  for(i in 1:n)
  {
    l=l+log(dlaplace(y[i], theta[1]+theta[2]*x[i],theta[3]))
    
  }
  return(-l)
}
theta=c(0.1,0.1,50)
fit=optim(theta,Negloglike,data=data)
##Estimated value of theta is:
c(fit$par[1],fit$par[2],fit$par[3])
```


(ii) Calculate **Bayesian Information Criterion** (BIC) for the model.



```{r warning=FALSE, include=TRUE}
BIC_B=ln(n)*(length(fit$par))+2*fit$value
#BIC value is:
BIC_B
```

**Part C**: We want to fit the following models:
$$
\texttt{Claims}_i\sim LogNormal(\mu_i,\sigma^2), where
$$
$$
\mu_i=\beta_0 + \beta_1 \log(\texttt{Holders}_i), ~~i=1,2,...,n
$$

Note that $\beta_0,\beta_1 \in\mathbb{R}$ and $\sigma \in \mathbb{R}^{+}$.

(i) Clearly write down the negative-log-likelihood function in `R`. Then use `optim` function to estimate MLE of $\theta=(\alpha,\beta,\sigma)$



```{r warning=FALSE, include=TRUE}

Negloglike=function(data,theta)
{
  l=0
  for(i in 1:n)
  {
    l=l+log(dlnorm(y[i], theta[1]+theta[2]*log(x[i]),theta[3]))
    
  }
  return(-l)
}

theta=c(0.1,0.1,1)
fit=optim(theta,Negloglike,data=data)
##Estimated value of theta is:

c(fit$par[1],fit$par[2],fit$par[3])

```

(ii) Calculate **Bayesian Information Criterion** (BIC) for the model.

```{r warning=FALSE, include=TRUE}
BIC_C=ln(n)*(length(fit$par))+2*fit$value
#BIC value is:
BIC_C

```




**Part D**: We want to fit the following models:
$$
\texttt{Claims}_i\sim Gamma(\alpha_i,\sigma), where
$$
$$
log(\alpha_i)=\beta_0 + \beta_1 \log(\texttt{Holders}_i), ~~i=1,2,...,n
$$

(i) Clearly write down the negative-log-likelihood function in `R`. Then use `optim` function to estimate MLE of $\theta=(\alpha,\beta,\sigma)$



```{r warning=FALSE, include=TRUE}

e=2.718281828459045
Negloglike=function(data,theta)
{
  l=0
  for(i in 1:n)
  {
    l=l+log(dgamma(y[i], e^(theta[1]+theta[2]*log(x[i])),theta[3]))
    
  }
  return(-l)
}

theta=c(0.1,0.1,0.1)
fit=optim(theta,Negloglike,data=data)

##Estimated value of theta is:

c(fit$par[1],fit$par[2],fit$par[3])


```

(ii) Calculate **Bayesian Information Criterion** (BIC) for the model.



```{r warning=FALSE, include=TRUE}

BIC_D=ln(n)*(length(fit$par))+2*fit$value
#BIC value is:
BIC_D
```


(iii) Compare the BIC of all three models
```{r warning=FALSE, include=TRUE}
c(BIC_A,BIC_B,BIC_C,BIC_D)
```
#

## Problem 5: Computational Finance - Modelling Stock prices

Following piece of code download the prices of TCS since 2007

```{r}
library(quantmod)
getSymbols('TCS.NS')
tail(TCS.NS)
```
Plot the adjusted close prices of TCS
```{r}
plot(TCS.NS$TCS.NS.Adjusted)
```

**Download the data of market index Nifty50**. The Nifty 50 index indicates how the over all market has done over the similar period.
```{r}
getSymbols('^NSEI')
tail(NSEI)
```
Plot the adjusted close value of Nifty50
```{r}
plot(NSEI$NSEI.Adjusted)
```


### Log-Return 
We calculate the daily log-return, where log-return is defined as
$$
r_t=\log(P_t)-\log(P_{t-1})=\Delta \log(P_t),
$$
where $P_t$ is the closing price of the stock on $t^{th}$ day.

```{r}
TCS_rt = diff(log(TCS.NS$TCS.NS.Adjusted))
Nifty_rt = diff(log(NSEI$NSEI.Adjusted))
retrn = cbind.xts(TCS_rt,Nifty_rt) 
retrn = na.omit(data.frame(retrn))
plot(retrn$NSEI.Adjusted,retrn$TCS.NS.Adjusted
     ,pch=20
     ,xlab='Market Return'
     ,ylab='TCS Return'
     ,xlim=c(-0.18,0.18)
     ,ylim=c(-0.18,0.18))
grid(col='grey',lty=1)
```

+ Consider the following model:

$$
r_{t}^{TCS}=\alpha + \beta r_{t}^{Nifty} + \varepsilon,
$$
where $\mathbb{E}(\varepsilon)=0$ and $\mathbb{V}ar(\varepsilon)=\sigma^2$.

1. Estimate the parameters of the models $\theta=(\alpha,\beta,\sigma)$ using the method of moments type plug-in estimator discussed in the class.

## Solution:
```{r}
mu_y = mean(retrn$TCS.NS.Adjusted)
mu_x = mean(retrn$NSEI.Adjusted)
sigma_y = sd(retrn$TCS.NS.Adjusted)
sigma_x = sd(retrn$NSEI.Adjusted)
rho = cor(retrn$NSEI.Adjusted, retrn$TCS.NS.Adjusted)
alpha_0 = mu_y - rho * mu_x * (sigma_x/sigma_y)
beta_0 = rho *  (sigma_x/sigma_y)
retrn_mom = retrn
retrn_mom$mom_est_TCS.NS.Adjusted = alpha_0 + beta_0 * retrn_mom$NSEI.Adjusted
epsilon = retrn_mom$TCS.NS.Adjusted-retrn_mom$mom_est_TCS.NS.Adjusted
sigma_0 = sd(epsilon)
theta_0 = c(alpha_0, beta_0, sigma_0)
print( theta_0)
```

2. Estimate the parameters using the `lm` built-in function of `R`. Note that `lm` using the OLS method.

## Solution:
```{r}
lm_model = lm(TCS.NS.Adjusted~NSEI.Adjusted, data = retrn)
co = lm_model$coefficients
alpha_1 = matrix(co)[1,1]
beta_1 = matrix(co)[2,1]
sigma_1 = sd(lm_model$residuals)
theta_1 = c(alpha_1, beta_1, sigma_1)
print(theta_1)
```


3. Fill-up the following table

Parameters | Method of Moments | OLS
-----------|-------------------|-----
$\alpha$   |                   |
$\beta$    |                   |
$\sigma$   |                   |

## Solution:

Parameters | Method of Moments | OLS
-----------|-------------------|-----
$\alpha$   |  $0.0005848199$   |$0.0004611208$
$\beta$    |  $0.3904739457$   |$0.7436970826$
$\sigma$   |  $0.0169171740$   |$0.0161865264$

4. If the current value of Nifty is 18000 and it goes up to 18200. The current value of TCS is Rs. 3200/-. How much you can expect TCS price to go up?

## Solution:

```{r}
prediction_mom = function(Nifty_initial_value, Nifty_final_value, TCS_initial_value){
  beta = theta_0[2]
  alpha = theta_0[1]
  x = log(Nifty_final_value) - log(Nifty_initial_value)
  y = alpha + beta*x
  y1 = log(TCS_initial_value) + y
  y2 = exp(y1)
  return(y2)
}
prediction_ols = function(Nifty_initial_value, Nifty_final_value, TCS_initial_value){
  beta = theta_1[2]
  alpha = theta_1[1]
  x = log(Nifty_final_value) - log(Nifty_initial_value)
  y = alpha + beta*x
  y1 = log(TCS_initial_value) + y
  y2 = exp(y1)
  return(y2)
}
```

By the Method of Moments type method, we can say that we can expect TCS value to go up by
```{r}
Nifty_initial_value = 18000
Nifty_final_value = 18200
TCS_initial_value = 3200
TCS_final_value = prediction_mom(Nifty_initial_value = Nifty_initial_value,
                                 Nifty_final_value = Nifty_final_value,
                                 TCS_initial_value = TCS_initial_value)
TCS_final_value - TCS_initial_value
```

By the OLS method, we can say that we can expect TCS value to go up by
```{r}
TCS_final_value = prediction_ols(Nifty_initial_value = Nifty_initial_value,
                                 Nifty_final_value = Nifty_final_value,
                                 TCS_initial_value = TCS_initial_value)
TCS_final_value - TCS_initial_value
```

Thus we can expect TCS price to go up by some value around Rs. 15.72 and Rs. 27.89.
