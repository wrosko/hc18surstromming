function pathLength = GetPathLength(path, cityLocation)
nCities = length(cityLocation);
pathLength = 0;
thisCity = path(nCities);
thisLocation = cityLocation(thisCity,:);
for i = 1:nCities
    nextCity = path(i);
    nextLocation = cityLocation(nextCity,:);
    distBetweenCities = norm(nextLocation - thisLocation);
    pathLength = pathLength + distBetweenCities;
    thisLocation = nextLocation;
end