pcd  = load('../data0/reproject_pcd.txt');
x = pcd(:,1);
y = pcd(:,2);
z = pcd(:,3);
scatter3(x,y,z);
% [X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),'v4'); %构造坐标点
% pcolor(X,Y,Z);
% shading interp;        %伪彩色图
% fcontourf(X,Y,Z);     %等高线图
% figure,surf(X,Y,Z);    %三维曲面