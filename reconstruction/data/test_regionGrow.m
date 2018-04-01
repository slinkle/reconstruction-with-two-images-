
img = imread('1.bmp');
I = rgb2gray(img); 
I = double(I);
x = 271;
y = 259;
J = My_RegionGrowing(I, [x, y], 0);
imshow(img)
hold on 
plot(y, x, 'p')
hold off
figure
imagesc(J)