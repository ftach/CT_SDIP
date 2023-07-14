function [x_filtered] = eliminate2(x,p)
%eliminate: Eliminates p non_zero porcentage of the x vector 
N = size(x, 2);
x_filtered = x; 
[x_sorted,x_position] = sort(abs(x), 'descend'); 
threshold_index = round(p * N) ;
x_filtered(x_position(threshold_index:end))=0; 
% threshold_value = x_sorted(threshold_index)
% for i=1:N
%     if abs(x(i))<threshold_value
%         x_filtered(i) = 0; 
%     end
% end

end

