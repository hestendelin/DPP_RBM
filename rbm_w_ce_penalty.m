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
maxepoch = 25;
epsilonw      = 0.1;   % Learning rate for weights 
epsilonvb     = 0.1;   % Learning rate for biases of visible units 
epsilonhb     = 0.1;   % Learning rate for biases of hidden units 
weightcost  = 0.0002;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;
ce_tracker = [];
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

 %%%%%%%%%%%%%%% Find classes %%%%%%%%%%%%%%%%%%%%%
 everyone = sum(sum(batchtargets,1),3);
 classes = find(everyone);
 num_of_classes = length(classes);
 hidden_units_per_class_cache = zeros(numhid,5000,10);
hidden_unit_y_neq_i_means = zeros(numhid,10);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 

for epoch = epoch:maxepoch,
    temp_ce_tracker = 0;
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 
 
 for idx=1:num_of_classes %%%%%% compute y!=i class means
     hidden_unit_y_neq_i_means(:,idx) = squeeze(mean(mean(hidden_units_per_class_cache(:,:,find(classes~=idx)),2),3));
 end
 hidden_units_per_class_cache = zeros(numhid,5000,10);
 cache_counter = ones(1,num_of_classes);
 
 for batch = 1:numbatches,
     
    % Find out how many per class in this batch
    num_per_class = zeros(1,num_of_classes);
    for case_idx = 1:numcases
        class_idx = find(batchtargets(case_idx,:,batch)==1);
        num_per_class(1,class_idx) = num_per_class(1,class_idx) + 1; 
    end
    num_per_class_max = max(num_per_class);
 
    y = zeros(numhid, num_per_class_max, num_of_classes);
    
 
%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data = batchdata(:,:,batch);
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));
  
  %%%%%%save hidden units for later in proper column form
  hidden_units = poshidprobs';
  class_idxs = ones(num_of_classes,1);
  for instance_idx=1:numcases
      class_var = find(batchtargets(instance_idx,:,batch)==1);
      y(:,class_idxs(class_var),class_var) = hidden_units(:,instance_idx);
      hidden_units_per_class_cache(:,cache_counter(class_var),class_var) = hidden_units(:,instance_idx);
      class_idxs(class_var) = class_idxs(class_var) + 1;
      cache_counter(class_var) = cache_counter(class_var) + 1;
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  
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

   
%%%%%%%%%% COMPUTE CROSS ENTROPY STATISTICS %%%%%%%%%%%%%%%%%% 
   
    tracker_sum = 0;
   ce_sum = 0;
   difference_differential = 0;
   for class_idx=1:num_of_classes
       temp_mean = hidden_unit_y_neq_i_means(:,class_idx);  %means from last epoch are used
       temp_y_eq_i = mean(y(:,:,class_idx),2);

       %temp_y_eq_i(find(temp_y_eq_i==0))=realmin;
       %temp_y_eq_i(find(temp_y_eq_i==1))=.999;
       
       if temp_y_eq_i-temp_mean==0
           temp_y_eq_i = temp_y_eq_i + .0001;
       end
       
       tracker_sum = tracker_sum + sum(power((1-abs(temp_y_eq_i - temp_mean)),2));
   
       ce_sum = ce_sum + (1./abs(temp_y_eq_i-temp_mean) - temp_y_eq_i + temp_mean);%.*temp_y_eq_i.*(1-temp_y_eq_i);
       %ce_sum(ce_sum==0) = realmin;
       %difference_differential = difference_differential + temp_y_eq_i - temp_mean;
   end
   
   if mod(batch,1000)==0 && epoch>1
    display(tracker_sum);
   end
   
   diversity_coeff = .00000001; %cross-validate this!  
   %difference_differential = difference_differential./10;
   ce_sum = ce_sum./10;
   
   if epoch==1
       div_grad_weights = 0;
       div_grad_bias = 0;
   else
       diversity_penalty_derivative = diversity_coeff * ce_sum;
       %temp_ce_tracker = temp_ce_tracker + sum(ce_sum);
       div_grad_weights = repmat(ce_sum',numdims,1);
       div_grad_bias = ce_sum';
   end

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid) + div_grad_weights;
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact) + div_grad_bias;

    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;

%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

 end
  ce_tracker = [ce_tracker temp_ce_tracker/(numbatches)];
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
end;
