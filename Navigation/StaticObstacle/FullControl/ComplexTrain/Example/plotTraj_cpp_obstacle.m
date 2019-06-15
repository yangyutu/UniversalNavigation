clear all
close all

nTarget = 7;
for i = 0 : nTarget
    data = load(['../testxyz_' num2str(i) '.txt']);
    dataSet{i+1} = data;
end

obsMap = load('../singleObstacle.txt');
[row, col] = find(obsMap);


figure(1)
for i = 1 : nTarget+1
hold on
x = dataSet{i}(:,2);
y = dataSet{i}(:,3);

plot(y,x, 'linewidth',1)
hold on
plot(col, row, 'square','markersize',8);
end
xlim([-1 20])
ylim([-1 20])
set(gca, 'ydir','reverse')

%set(gca,'box','off')
%set(gca,'visible','off')
set(gca,'linewidth',2,'fontsize',20,'fontweight','bold','plotboxaspectratiomode','manual','xminortick','on','yminortick','on');
set(gca,'TickLength',[0.04;0.02]);
pbaspect([1 1 1])
%saveas(gcf,'traj.png')