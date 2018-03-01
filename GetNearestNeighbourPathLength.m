function pathLength = GetNearestNeighbourPathLength(nCars, tMax, dij, si, fi, b, wi)
nNodes = size(dij,1);
tabuList = [];

timeCarAvailable = zeros(nCars);
carLocations = zeros(nCars) + nNodes;
carPaths = cell(1,nCars);
for c = 1:nCars
    carPaths{c} = [];

for t = 0:tMax
    avail = find(timeCarAvailable<=t);
    if length(avail)==0
        continue
    end
    availCarLocations = carLocations(avail);
    dcarj = dij(availCarLocations,:);
    etaij = GetVisibility(nNodes, t, si, dcarj, wi, b, fi);
    for car = 1:length(avail)
        [m, m_ind] = max(etaij(car,:));
        carLocations(car) = m_ind;
        tabuList = [tabuList m_ind];
        carPaths{car} = [carPaths{car} m_ind];
        timeCarAvailable(car) = timeCarAvailable(car) + dcarj(car,m_ind) + wi(m_ind);
    end
end

