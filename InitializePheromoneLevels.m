function pheromoneLevel = InitializePheromoneLevels(nCities, tau0)
pheromoneLevel = zeros(nCities) + tau0;
for diagonalElement = 1:nCities
    pheromoneLevel(diagonalElement, diagonalElement) = 0;
end