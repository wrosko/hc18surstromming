function path = GeneratePath(pheromoneLevel, distgraph, starttimes, lengths, bonus, ncars, maxtime, alpha, beta)
nCities = length(lengths);
startCity = nCities;
notTabuList = ones(1, nCities);
isDriving = zeros(ncars,1);
arrival = zeros(ncars,1);

notTabuList(startCity) = 0;
thisCity = ones(nCars,1) * startCity;


path = {};
for car = 1:ncars
    path(car) = [startCity];
end

for t = 1:maxtime
    for car = 1:ncars
        if arrival(car) < t
            isDriving(car) = 0;
        end
    end
    
    indizes = randperm(numel(isDriving));
    cars = isDriving(indizes);
    visibility = GetVisibility(nCities, t, starttimes(:,1), distgraph(thisCity(indizes),:), ...
        lengths, bonus,starttimes(:,2)) ;
    for car = cars
        remainingCities = find(notTabuList); % selects cities which are not in tabuList
        tauIJalpha = power(pheromoneLevel(thisCity(car), remainingCities), alpha);
        if len(visibility) > 0
            etaIJbeta = power(visibility, beta);
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
            notTabuList(nextCity) = 0;
            arrival(car) = t ...
                + distgraph(thisCity(car),nextCity) ... %driving time
                + max(0,starttimes(nextCity,1) - t - lengths(thisCity(car))) ... %waiting time
                + lengths(nextCity); %driving time 2
            thisCity(car) = nextCity;
            path(car) = [path(car), nextCity];
        end
    end
end
