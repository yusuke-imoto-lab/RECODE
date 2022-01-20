RECODE <- function(
  X
){
  message('--START RECODE--')
  message(' I.   Normalizing')
  X_clean <- t(as.matrix(X))[,apply(X,1,sum)>0]
  n <- nrow(X_clean)
  d <- ncol(X_clean)
  X_nUMI <- apply(X_clean,1,sum)
  X_prob <- X_clean/X_nUMI
  X_prob_mean <- apply(X_prob,2,mean)
  noise_var <- apply(X_clean/X_nUMI/X_nUMI,2,mean)
  noise_var[noise_var==0] = 1
  X_norm <- t((t(X_prob)-X_prob_mean)/sqrt(noise_var))
  message(' II.  Projecting to PCA space')
  pca = prcomp(X_norm,scale=F)
  message(' III. Modifying eigenvalues')
  eigval = pca$sdev**2
  n_eigval = length(eigval)
  U = pca$rotation
  eigval_mod <- eigval
  for (i in 1:length(eigval)-1) eigval_mod[i] <- eigval[i]-mean(eigval[(i+1):length(eigval)])
  eigval_mod[length(eigval_mod)] = 0
  eigval_sum <- eigval
  for (i in 1:length(eigval)) eigval_sum[i] <- sum(eigval[i:length(eigval)])
  threshold = sum(X_prob_mean>0)-0:(length(eigval)-1)
  ell_func = eigval_sum - threshold
  ell = sum(ell_func>0)+1
  message('      ell = ',ell)
  Lam = diag(sqrt(eigval_mod[1:ell]/eigval[1:ell]))
  message('   Reducing noise')
  X_norm_RECODE <- X_norm %*% U[,1:ell] %*% Lam %*% t(U[,1:ell])
  message(' VI.  Reversing to original space')
  X_prob_RECODE <- t(t(X_norm_RECODE)*sqrt(noise_var)+X_prob_mean)
  X_RECODE_clean = X_prob_RECODE*X_nUMI
  X_RECODE_clean[X_RECODE_clean<0] = 0
  X_RECODE <- t(X)
  X_RECODE[,apply(X,1,sum)>0] <- X_RECODE_clean
  message('--END RECODE--')
  return(t(X_RECODE))
}
