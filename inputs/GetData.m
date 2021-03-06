clear all;

filename = "a_example";
%filename = "b_should_be_easy";
%filename = "c_no_hurry";
%filename = "d_metropolis";
%filename = "e_high_bonus";
[distgraph, starttimes, lengths, bonus, ncars, maxtime] = DataHelper(strcat(filename,'.in'));
save(strcat(filename,'.data'),'distgraph', 'starttimes', 'lengths', 'bonus', 'ncars', 'maxtime');


function [distgraph, starttimes, lengths, bonus, ncars, maxtime] = DataHelper(filename)
    data = importdata(filename);
    header = data(1,:);
    ncars = header(3);
    nrides = header(4)+1;
    bonus = header(5);
    maxtime = header(6);
    
    data = data(2:end,:);
    data = [data; [0,0,0,0,0,0]];
    distgraph = zeros(nrides,nrides,'int32');
    starttimes = zeros(nrides,2,'int32');
    lengths = zeros(nrides,1,'int32');
    
    for i = 1:nrides
        ride1 = data(i,:);
        lengths(i) = GetDistance(ride1, ride1);
        starttimes(i,1) = ride1(5);
        starttimes(i,2) = ride1(6) - lengths(i);
        distgraph(i,i) = NaN; 
        for j = 1:i
            ride2 = data(j,:);
            distgraph(i,j) = GetDistance(ride1, ride2);
            distgraph(j,i) = GetDistance(ride2, ride1);
        end
    end
end

function d = GetDistance(ride1, ride2)
    x1 = ride1(3);
    y1 = ride1(4);
    x2 = ride2(1);
    y2 = ride2(2);
    d = abs(x1 - x2) + abs(y1 - y2);
end