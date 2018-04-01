clc;

first_2 = imread('first_2.tif');

m = zeros(size(first_2,1), size(first_2,2));
m(50:100, 50:150) = 1;
seg_img = chenvese(first_2,m,500,0.1,'chan');