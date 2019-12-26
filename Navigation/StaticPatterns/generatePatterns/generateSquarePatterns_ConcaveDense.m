clear all
close all



shape1 =     [
        [ 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        [ 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [ 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [ 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [ 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [ 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [ 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [ 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        
    ];


s = 10; 

d = 8;

y0 = 0.5 * s;
y = [];
x = [];
n = 4;
for i = 1: n
    y = [y; (i - 1) * (2*s + 2 * d)];
    x = [x; 0];
end

% middle lane
for i = 1: n - 1
    y = [y; (i - 1) * (2*s + 2 * d) + d + s];
    x = [x; d];
end

for i = 1: n
    y = [y; (i - 1) * (2*s + 2 * d)];
    x = [x; d + s];
end

figure(1)

y = y + s/2 + 1;
x = x + s/2 + 1;
plot(y + s/2, x + s / 2, 'o');



figure(2)

map = zeros(max(x) - min(x) + s, max(y) - min(y) + s);

for i = 1: length(x)
    map([(x(i) - s/2): (x(i) + s/2 - 1) ], [(y(i) - s/2): (y(i) + s/2 -1)]) = shape1;
end



map =  padarray(map,[2 2],1,'both');
imshow(map)
dlmwrite('patternMap_concavedense.txt', map, 'delimiter',' ', 'precision', '%d');
