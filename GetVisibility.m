% function visibility = GetVisibility(cityLocation)
% nCities = length(cityLocation);
% visibility = zeros(nCities);
% for i = 1:nCities-1
%     for j = i:nCities
%         separationVector = cityLocation(i,:) - cityLocation(j,:);
%         viz = 1/norm(separationVector);
%         visibility(i,j) = viz;
%         visibility(j,i) = viz;
%     end
% end
function visibility = GetVisibility(nJobs, t, s_i, d_carj, W_i, b, f_i)
nCars = size(d_carj,1);
visibility = zeros(nCars,nJobs);
wait_at_dest = WaitAtDestination(s_i, d_carj, t);


for carN = 1:nCars
    lateCheck = f_i - d_carj(carN,:)' - t;
    greater_than_zero = (lateCheck >= 0);
    current_wait = wait_at_dest(:,carN);
    
    b_i = (current_wait >= 0) * b;
    eta = (W_i + b_i) ./ (d_carj .* (abs(current_wait) + 1));
    eta = greater_than_zero * eta;
    visibility(car,:) = eta;
end
end