clear all
close all

nTarget = 0;

data = load(['testxyz_' num2str(nTarget) '.txt']);

obsMap = load('ultraComplexTrainSmall.txt');
[row, col] = find(obsMap);


figure(1)
hold on
x = data(:,2);
y = data(:,3);

plot(y,x, 'linewidth',1)
plot(y(1), x(1),'o')
plot(y(end), x(end),'square')
hold on
plot(col, row, 'square','markersize',8);

xlim([-1 80])
ylim([-1 80])
set(gca, 'ydir','reverse')

%set(gca,'box','off')
%set(gca,'visible','off')
set(gca,'linewidth',2,'fontsize',20,'fontweight','bold','plotboxaspectratiomode','manual','xminortick','on','yminortick','on');
set(gca,'TickLength',[0.04;0.02]);
pbaspect([1 1 1])
%saveas(gcf,'traj.png')