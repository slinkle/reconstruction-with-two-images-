pcd  = load('../data0/reproject_pcd.txt');
x = pcd(:,1);
y = pcd(:,2);
z = pcd(:,3);
scatter3(x,y,z);
% [X,Y,Z]=griddata(x,y,z,linspace(min(x),max(x))',linspace(min(y),max(y)),'v4'); %���������
% pcolor(X,Y,Z);
% shading interp;        %α��ɫͼ
% fcontourf(X,Y,Z);     %�ȸ���ͼ
% figure,surf(X,Y,Z);    %��ά����