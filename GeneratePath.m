function path = GeneratePath(pheromoneLevel, visibility, alpha, beta)
nCities = length(pheromoneLevel);
startCity = randi(nCities);
notTabuList = ones(1, nCities);

tabuList = [startCity];
notTabuList(startCity) = 0;
thisCity = startCity;

for city = 1:nCities-1
    remainingCities = find(notTabuList); % selects cities which are not in tabuList
    tauIJalpha = power(pheromoneLevel(thisCity, remainingCities), alpha);
    etaIJbeta = power(visibility(thisCity, remainingCities), beta);
    numerator = tauIJalpha.*etaIJbeta;
    denominator = sum(numerator);
    probabilityOfTraversal = numerator/denominator;
    
    r = rand();
    indexOfCity = 0;
    while r > 0
        r = r - probabilityOfTraversal(1);
        probabilityOfTraversal(1) = [];
        indexOfCity = indexOfCity + 1;
    end
    
    nextCity = remainingCities(indexOfCity);
    tabuList = [tabuList nextCity];
    notTabuList(nextCity) = 0;
    thisCity = nextCity;
    
end

path = tabuList;