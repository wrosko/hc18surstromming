function deltaPheromoneLevel = ComputeDeltaPheromoneLevels(pathCollection, totalScoreCollection,ridesNumber,nCar)

deltaPheromoneLevel = zeros(ridesNumber);

for iSimulation = 1:length(C) % for Fabian this is for each ant
    thisPathScore = totalScoreCollection(iSimulation); %Fixed score for each simulation
    actualPathCollection = pathCollection{iSimulation};
    for iCar = 1:nCar
        thisPath = actualPathCollection{iCar};
        origin = thisPath(end); 
        for loc = 1:length(thisPath)
            dest = thisPath(loc);
            deltaPheromoneLevel(dest, origin) = deltaPheromoneLevel(dest, origin) + thisPathScore;
            origin = dest;
        end
        deltaPheromoneLevel = deltaPheromoneLevel - diag(diag(deltaPheromoneLevel)); % Put diag elements = 0
    
    end
end