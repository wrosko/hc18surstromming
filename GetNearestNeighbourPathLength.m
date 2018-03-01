function pathLength = GetNearestNeighbourPathLength(cityLocation)
nCities = size(cityLocation, 1);

distanceMatrix = zeros(nCities) * NaN;
for i = 1:(nCities-1)
    for j = (i+1):nCities
        vectBetweenCities = cityLocation(i,:) - cityLocation(j,:);
        distBetweenCities = norm(vectBetweenCities);
        distanceMatrix(i,j) = distBetweenCities;
        distanceMatrix(j,i) = distBetweenCities;
    end
end

NNPath = zeros(1, nCities);
pathLength = 0;
firstCity = randi([1 nCities]);
NNPath(1) = firstCity;
distanceMatrix(:, firstCity) = NaN(1, size(distanceMatrix, 1));
currentCity = firstCity;
for i = 2:nCities
    distances = distanceMatrix(currentCity, :);
    [distToNearest, nearestCity] = min(distances);
    pathLength = pathLength + distToNearest;
    NNPath(i) = nearestCity;
    currentCity = nearestCity;
    distanceMatrix(:, currentCity) = NaN(1, size(distanceMatrix, 1));
end
vectLastToFirst = cityLocation(firstCity, :) - cityLocation(currentCity, :);
distLastToFirst = norm(vectLastToFirst);
pathLength = pathLength + distLastToFirst;