function visibility = GetVisibility(cityLocation)
nCities = length(cityLocation);
visibility = zeros(nCities);
for i = 1:nCities-1
    for j = i:nCities
        separationVector = cityLocation(i,:) - cityLocation(j,:);
        viz = 1/norm(separationVector);
        visibility(i,j) = viz;
        visibility(j,i) = viz;
    end
end