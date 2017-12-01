function [ LL ] = loglikelihood( theta, y, X )
% LOGLIKEHIHOOD - computes the log of the likelihood function for OLS under normal errors

betas = theta(1:end-1)'; % coefficients
sigma_epsilon = theta(end); % sd of the error term

epshat = y - X*betas; % vector of residuals under current parameters

ll_obs = -log(sigma_epsilon^2) - 0.5 * log(2 * pi) - ...
    (epshat .^ 2) ./  (2 * sigma_epsilon^2); % log likelihood of all individual observations

LL = sum(ll_obs);

end

