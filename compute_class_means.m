function [ means_of_other_classes ] = compute_class_means( num_of_classes )
%computes the means of the other classes

numPerClass = 5000;
num_of_vis_dims = 784;
class_means = nan(num_of_vis_dims,num_of_classes);
for idx=1:num_of_classes
    load(['digit' num2str(idx-1) '.mat']);
    class_means(:,idx) = mean(D(1:numPerClass,:))';
end

means_of_other_classes = nan(num_of_vis_dims, num_of_classes);
class_array = 1:num_of_classes;
for idx=1:num_of_classes
   means_of_other_classes(:,idx) = mean(class_means(:,find(class_array~=idx)),2); 
end

means_of_other_classes = means_of_other_classes ./ 255;

end
