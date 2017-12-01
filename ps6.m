%%%%%%%%% hw 6
clear
%%% import data
import_card_data
y = lwage;
X = [educ exper south black smsa ones(length(y), 1)];

%%% run regular OLS
[beta_OLS, OLS_intervals, ~, ~, OLS_stats] = regress(y,X);
OLS_se = (OLS_intervals(:,2) - OLS_intervals(:,1)) / (2*1.96);
OLS_resid_sd = sqrt(OLS_stats(4));

%%% do MCMC
% setup:
burnin = 10*1000;
N = 50*1000 + burnin;
theta_init = [beta_OLS', OLS_resid_sd];
k = length(theta_init); % k = number of parameters
theta_sequence = zeros(N, k); % setup sample chain for parameters
theta_sequence(1,:) = theta_init;
llhs = zeros(N,1); % store sequence of loglikelihoods
accept = zeros(N-1,1); % keep track of acceptances

% set up likelihood and prior
logprior = @(theta) -(theta(1)-0.08).^2./(2*0.03.^2);
% likelihood function is in loglikelihood.m

% specify std for proposal steps
scale_factor = 0.125;
proposal_stds = scale_factor * [OLS_se', 0.05];

llhs(1) = logprior(theta_init) + loglikelihood(theta_init, y, X);

rng(0)
for t = 2:N
    % proposal step
    theta = theta_sequence(t-1,:); % previous sample point
    theta_prop = theta + proposal_stds .* randn(1,k);
    proposal_llh = logprior(theta_prop) + loglikelihood(theta_prop, y, X);
    
    % accept / reject
    crit_value = log( rand(1) );
    if proposal_llh - llhs(t-1) > crit_value
        % accept parameter proposal
        theta_sequence(t,:) = theta_prop;
        llhs(t) = proposal_llh;
        accept(t-1) = 1;
    else
        % reject parameter proposal
        theta_sequence(t,:) = theta_sequence(t-1,:);
        llhs(t) = llhs(t-1);
        accept(t-1) = 0;
    end
    
    
    if mod(t,5000) == 0
        waitbar(t/N)
        display(mean(accept(t-4999:t-1)))
        display(theta)
    end
    
    
end

posterior_means = mean(theta_sequence(burnin+1:end));
acceptance_rate = mean(accept(burnin+1:end));
for ii=1:k
    figure
    hist(theta_sequence(burnin+1:end,ii))
end

