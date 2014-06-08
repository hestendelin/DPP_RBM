function Z = rbm_partfun_ais(W,a,b,nSteps,nSamps)
% Approximate the partition function of a trained RBM using AIS

% Compute the partition function ratio between the "base-rate" RBM (this
% RBM's visible unit bias, but with zero weights and hidden function
% biases) and this RBM
partRat = rbm_ais(zeros(size(W)),b,zeros(size(a)),W,b,a,nSteps,nSamps);

% compute the partition function of the base-rate RBM
Za = 2^(size(a))*prod(1+exp(b));

% The partition function of the original RBM is now a simple function of
% the base-rate partition function and the partition function ratio
Z = Za*partRat;
end