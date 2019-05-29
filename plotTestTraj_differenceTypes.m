clear all
close all

addpath('')
%addpath(genpath('E:\Dropbox\matlabscript\'))
fileNames = {'trajOutCIRCLERxyz_0.txt', 'trajOutFULLCONTROLxyz_0.txt', 'trajOutSLIDERxyz_0.txt', 'trajOutVANILLASPxyz_0.txt'};

for i =1:length(fileNames)
data = load(fileNames{i});

x = data(:,2);
y = data(:,3);

figure(1)
hold on
plot(x, y)
end
legend('circler','full','slider','sp')

   