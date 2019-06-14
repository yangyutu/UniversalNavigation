clear all
close all

addpath('')
%addpath(genpath('E:\Dropbox\matlabscript\'))
data = load('Traj/testxyz_0.txt');
np = 1;
N=1;
a=1e-6;
k=1.38e-23;
T=293.15;
dt = 1;
x = data(:,2);
y = data(:,3);

phi = data(:,4);
u = data(:,5);
nframe = length(x);
mvflag = 1;

x= reshape(x,N,nframe);
y =reshape(y,N,nframe);
phi = reshape(phi,N,nframe);

u = reshape(u,N,nframe);

for i = 1:N
    figure(1)
    hold on
    plot(x(i,:), y(i,:))
end
xlim([10, 20])
ylim([10, 20])

