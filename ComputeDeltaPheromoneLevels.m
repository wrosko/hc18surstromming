function deltaPheromoneLevel = ComputeDeltaPheromoneLevels(pathCollection, pathLengthCollection)
nPaths = size(pathCollection, 1);
nCities = size(pathCollection, 2);
deltaPheromoneLevel = zeros(nCities);

for iPath = 1:nPaths
    thisPath = pathCollection(iPath, :);
    thisPathLength = pathLengthCollection(iPath);
    origin = thisPath(nCities);
    for loc = 1:nCities
        dest = thisPath(loc);
        deltaPheromoneLevel(dest, origin) = deltaPheromoneLevel(dest, origin) + 1/thisPathLength;
        origin = dest;
    end
end