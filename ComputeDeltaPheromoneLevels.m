function deltaPheromoneLevel = ComputeDeltaPheromoneLevels(pathCollection, totalScoreCollection)
%{
pathCollection:
Each row contains the indexes of the ride. 
If a car is moving from one node to another, just put it equal to the starting. 
If it has finished to move, than it will just repeat the last cell.

totalScoreCollection:
Total sum of (w+b) scored by that car
%}

nCar = size(pathCollection, 1);
nMaxRide = size(pathCollection, 2);
deltaPheromoneLevel = zeros(nMaxRide);

for iCar = 1:nCar
    thisPath = pathCollection(iCar, :);
    thisPathScore = totalScoreCollection(iCar);
    origin = thisPath(nMaxRide); % Assuming that if the car dsnt move than same value in the next cell
    for loc = 1:nMaxRide
        dest = thisPath(loc);
        deltaPheromoneLevel(dest, origin) = deltaPheromoneLevel(dest, origin) + thisPathScore;
        origin = dest;
    end
    deltaPheromoneLevel = deltaPheromoneLevel - diag(diag(deltaPheromoneLevel)); % Put diag elements = 0
end