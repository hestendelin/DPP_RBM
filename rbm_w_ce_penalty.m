% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 

numhid = 500;
restart = 1;
maxepoch = 50;
epsilonw      = 0.1;   % Learning rate for weights 
epsilonvb     = 0.1;   % Learning rate for biases of visible units 
epsilonhb     = 0.1;   % Learning rate for biases of hidden units 
weightcost  = 0.0002;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;
dpp_tracker = [];
[numcases numdims numbatches]=size(batchdata);

if restart ==1,
  restart=0;
  epoch=1;

% Initializing symmetric weights and biases. 
  vishid     = 0.1*randn(numdims, numhid);
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);

  poshidprobs = zeros(numcases,numhid);
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end


for epoch = epoch:maxepoch,
    temp_dpp_tracker = 0;
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches,
 %fprintf(1,'epoch %d batch %d\r',epoch,batch); 

    classes = [];
    for instance_idx = 1:numcases
        classes = [classes find(batchtargets(instance_idx,:,batch)==1)];
    end
    num_of_classes = size(unique(classes),2);
    num_per_class = zeros(1,num_of_classes);
    for class_idx = 1:size(classes,2)
        num_per_class(1,classes(class_idx)) = num_per_class(1,classes(class_idx)) + 1; 
    end
    num_per_class_max = max(num_per_class);
 
    y = zeros(numhid, num_per_class_max, num_of_classes);

 
%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));
  hidden_units = poshidprobs';
  for instance_idx=1:numcases
      class_var = find(batchtargets(instance_idx,:,batch)==1);
      y(:,class_idx,find(batchtargets(instance_idx,:,batch)==1)) = hidden_units(:,instance_idx);
  end
    %%%%%%save hidden units for later in proper column form
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;
  poshidact   = sum(poshidprobs);
  posvisact = sum(data);

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  poshidstates = poshidprobs > rand(numcases,numhid);

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    
  negprods  = negdata'*neghidprobs;
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  err= sum(sum( (data-negdata).^2 ));
  errsum = err + errsum;

   if epoch>5,
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end;

   %%%%COMPUTE DPP NUMBERS%%%%
   %normalize columns so we have a true probability (effect of this?)
   ce_stats = zeros(num_hid,num_of_classes);
   for class_idx=1:num_of_classes
       ce_stats(:,class_idx) = mean(y(:,:,class_idx),2);
   end
   ce_sum = 0;
   difference_differential = 0;
   class_idx_list = [1:num_of_classes];
   for class_idx=1:num_of_classes
       temp_mean = mean(ce_stats(:,find(class_idx_list~=class_idx)),2);
       ce_sum = ce_sum + sum( -1.*temp_mean.*log(ce_stats(:,class_idx)) - (1-temp_mean).*log(1-ce_stats(:,class_idx)));
       difference_differential = difference_differential + sum(ce_stats(:,class_idx) - temp_mean);
   end
   gram = hidden_units'*hidden_units;
   dpp_prob = det(gram);
   temp_dpp_tracker = temp_dpp_tracker + dpp_prob;
   if mod(batch,1000) == 0
       display(dpp_prob);
   end
   diversity_coeff = .005; %cross-validate this!  
   
   diversity_penalty_derivative = diversity_coeff * (-1/ce_sum^2) * difference_differential;


%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid) - diversity_penalty_derivative;
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact) - diversity_penalty_derivative;

    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;

%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

 end
  dpp_tracker = [dpp_tracker temp_dpp_tracker/(numbatches)];
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
end;
