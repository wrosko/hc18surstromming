function path = GeneratePath(pheromoneLevel, distgraph, starttimes, lengths, bonus, ncars, maxtime, alpha, beta)
nCities = length(lengths);
startCity = nCities;
notTabuList = ones(1, nCities);

tabuList = [startCity];
notTabuList(startCity) = 0;
thisCity = startCity;

for t = 1:
    for car = 1:ncars
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
end

path = tabuList;