%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Ant system (AS) for TSP.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clear all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load('a_example.data') % loads distgraph, starttimes, lengths, bonus, ncars
nRides = length(lengths);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numberOfAnts = nRides*2;  
alpha = 1;
beta = 4;
rho = 0.2;

nearestNeighbourPathLength = GetNearestNeighbourPathLength(cityLocation);
tau0 = numberOfAnts/nearestNeighbourPathLength;

targetPathLength = 121.1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

range = [0 20 0 20];
tspFigure = InitializeTspPlot(cityLocation, range);
connection = InitializeConnections(cityLocation);
pheromoneLevel = InitializePheromoneLevels(numberOfCities, tau0);
visibility = GetVisibility(cityLocation);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
minimumPathLength = inf;

iIteration = 0;
pathCollection = zeros(numberOfAnts, numberOfCities);
pathLengthCollection = zeros(numberOfAnts,1);

while (minimumPathLength > targetPathLength)
 iIteration = iIteration + 1;

 %%%%%%%%%%%%%%%%%%%%%%%%%%
 % Generate paths:
 %%%%%%%%%%%%%%%%%%%%%%%%%%

 for k = 1:numberOfAnts
  path = GeneratePath(pheromoneLevel, distgraph, starttimes, lengths, bonus, ncars, alpha, beta);
  pathLength = GetPathLength(path,cityLocation);
  if (pathLength < minimumPathLength)
    bestPath = path;
    minimumPathLength = pathLength;
    disp(sprintf('Iteration %d, ant %d: path length = %.5f',iIteration,k,minimumPathLength));
  %  PlotPath(connection,cityLocation,path);
  end
  pathCollection(k,:) = path;
  pathLengthCollection(k) = pathLength;
 end

 %%%%%%%%%%%%%%%%%%%%%%%%%%
 % Update pheromone levels
 %%%%%%%%%%%%%%%%%%%%%%%%%%

 deltaPheromoneLevel = ComputeDeltaPheromoneLevels(pathCollection,pathLengthCollection,nRides,ncars);
 pheromoneLevel = UpdatePheromoneLevels(pheromoneLevel,deltaPheromoneLevel,rho);

end

%saveas(tspFigure, "21cACO.png", 'png');