function deltaPheromoneLevel = ComputeDeltaPheromoneLevels(pathCollection, totalScoreCollection,ridesNumber)

nCar = length(pathCollection);
deltaPheromoneLevel = zeros(ridesNumber);

for iCar = 1:nCar
    thisPath = cell2mat(pathCollection(iCar));
    thisPathScore = totalScoreCollection(iCar);
    origin = thisPath(end); 
    for loc = 1:length(thisPath)
        dest = thisPath(loc);
        deltaPheromoneLevel(dest, origin) = deltaPheromoneLevel(dest, origin) + thisPathScore;
        origin = dest;
    end
    deltaPheromoneLevel = deltaPheromoneLevel - diag(diag(deltaPheromoneLevel)); % Put diag elements = 0
end