function partRat = rbm_ais(Wa,ba,aa,Wb,bb,ab,nSteps,nSamples)
% estimate the partition function ratio between two RBMs

sigmoid = @(x) 1./(1+exp(-x));
betaKs = linspace(0,1,nSteps);

wts = zeros(1,nSamples);

% Create nSamples chains of samples
for samp = 1:nSamples
    %keep track of all of the visible units so we can compute our weight
    %for this chain
    v = zeros(nSteps,length(ba));
    probRats = zeros(nSteps-1,1);
    % for our first visible unit, we randomly initialize the hidden units
    % and then sample from the RBM A.
    hA = rand(length(aa),1);
    % sample each element of the visible unit
    v(1,:) = sigmoid(Wa*hA + ba);
    
    
    % Now, get the rest of the samples
    for i=2:nSteps
        % sample a hidden unit for both RBMs using the previous v
        % not sure about this line...
        hA = sigmoid((1-betaKs(i))*(Wa'*v(i-1,:)'+aa));
        hB = sigmoid(betaKs(i)*(Wb'*v(i-1,:)' + ab));
        
        % now we can sample our new v
        v(i,:) = sigmoid((1-betaKs(i))*(Wa*hA+ba) + ...
            betaKs(i)*(Wb*hB+bb));
        % get the probability ratio
        probRats(i-1) = marginalK(v(i,:)',betaKs(i),Wa,ba,aa,Wb,bb,ab)/...
            marginalK(v(i,:)',betaKs(i-1),Wa,ba,aa,Wb,bb,ab);
    end
    wts(samp) = prod(probRats);
end
% Our estimate of the ratio of the partition functions is just the mean of
% he importance weights
partRat = mean(wts);
end


function prob = marginalK(v,betaK,Wa,ba,aa,Wb,bb,ab)
% Define the two big products we have to calculate, then multiply
% everything together
prodA = 1;
prodB = 1;
for j=1:length(aa)
    prodA = prodA * (1 + exp((1-betaK)*(Wa(:,j)'*v+aa(j))));
end
for j=1:length(ab)
    prodB = prodB * (1 + exp(betaK*(Wb(:,j)'*v+ab(j))));
end
prob = exp((1-betaK)*(ba'*v))*prodA*exp(betaK*(bb'*v))*prodB;
end