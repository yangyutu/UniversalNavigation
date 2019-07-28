clear all
close all

data = load('../testxyz_0.txt');
obsData = load('../testdynamicObs_0.txt');
x = data(:,2);
y = data(:,3);

obsx = obsData(:,3);
obsy = obsData(:,4);
obsPhi = obsData(:,5);
Nobs = max(obsData(:,2)) + 1;
nframe = length(x);
obsx = reshape(obsx, Nobs, nframe);
obsy = reshape(obsy, Nobs, nframe);
obsPhi = reshape(obsPhi, Nobs, nframe);
polygon = [0 0;
           0 2;
           3 4;
           3 7;
           0 9;
           0 11;
           2 11;
           4 8;
           7 8;
           9 11;
           11 11;
           11 9;
           8 7;
           8 4;
           11 2;
           11 0;
           9 0;
           7 3;
           4 3;
           2 0;
           0 0;];
polygon = polygon - 6;
figure(5)
fill(polygon(:,1), polygon(:,2),'r')

figure(1)


plot(x, y)
xlim([0 120])
ylim([0 40])
hold on

for i = 1: Nobs
    plot(obsx(i,:), obsy(i,:));
    hold on
end

movieFlag = 1;
if movieFlag
    skip = 10;
    for j = 1:skip:nframe
    figure(2)
    hold on
    plot(x(j), y(j),'o','markersize',12)
xlim([0 120])
ylim([0 40])
hold on

for i = 1: Nobs
    phi = obsPhi(i,j);
    rotMat = [cos(phi) sin(phi); -sin(phi) cos(phi)];
    polygonRot = (rotMat * polygon')';
    fill(obsx(i,j) + polygonRot(:,1), obsy(i,j) + polygonRot(:,2),'r');
    hold on
end
pbaspect([3,1,1]);
saveas(gcf,['rendering_' num2str(j) '.png'])
    close(2)

    end
end
    