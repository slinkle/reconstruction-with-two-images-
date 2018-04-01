clc;
close all;
clear ;

% �������һ�飨x,y,z),��Щ���������һ���ռ�ƽ��ȽϽ�
% x0=1;L1=2;
% y0=1;L2=2;
% x=x0+rand(20,1)*L1;
% y=y0+rand(20,1)*L2;
% z=1+2*x+3*y;
% pcd = load('reproject_pcd.txt');
%%%--------------------------------------------------------------------%%%%
% ���ص�������
pcd = load('xyzw_pcd.txt');
% pcd(:,3) = -pcd(:,3);
x = pcd(:,1); y = pcd(:,2); z = pcd(:,3); 
% w = pcd(:,4);
figure(1);
scatter3(x,y,z,'filled')
hold on;
X = [ones(length(x),1) x y];

% ��ϣ���ʵ�����Իع飬�������������ƽ��
% ���Ϊ b = [b(1) b(2) b(3)] ��ʾ z = b(1) + b(2)*x + b(3)*y ����ϳ�����ƽ��ķ���
[b,bint,r,rint,stats] = regress(z,X,90);


%%%--------------------------------------------------------------------%%%%
% ͼ�λ���
xfit = min(x):0.001:max(x);
yfit = min(y):0.001:max(y);
[XFIT,YFIT]= meshgrid (xfit,yfit);
ZFIT = b(1) + b(2) * XFIT + b(3) * YFIT;
mesh(XFIT,YFIT,ZFIT);

%%%--------------------------------------------------------------------%%%%
% �������ϵĵ�ͶӰ��ԭͼ���д���
Q_param = load('Q.txt');
if size(Q_param,2) ~= 4
    fprintf('Q file error\n');
end
Q = [1 0 0 Q_param(1);
     0 1 0 Q_param(2);
     0 0 0 Q_param(3);
     0 0 Q_param(4) 0];
img = ones(600,800);

%%%--------------------------------------------------------------------%%%%
% points = zeros(length(x),3);
high_point = zeros(1,size(pcd,2));
max_dist = 0;
min_dist = 0.02;
points = [];
ground = [];
for index = 1:length(x)
    % ����㵽��ľ���
    norm_ = double(sqrt((b(2)*b(2) + b(3)*b(3)+ (-1)*(-1))));
    dist = abs(b(2)*x(index) + b(3)*y(index) - z(index) + b(1)) / norm_;
    % �㵽��ľ������2cm����Ϊʯͷ�ϵĵ�
    if (dist > min_dist)
        points = [points; pcd(index,:)];
        if dist > max_dist
            max_dist = dist;
            high_point = pcd(index,:);
        end
    else
%         ground = [ground; pcd(index,:)];
        ground = pcd(index,:);
        ground(1:3) = ground(1:3)*ground(4);
        img_data = Q\ground';
        img(round(img_data(2)),round(img_data(1))) = 0;
    end
end


%%%--------------------------------------------------------------------%%%%
% ʯͷ������: (midx,midy,midz)
% (midx,midy,maxz)->ͶӰ��ƽ��ĵ�(zerox,zeroy,zeroz)
% midz = (maxz + z')/2
% ʯͷ��ֱ����dist((midx,midy,maxz),(zerox,zeroy,zeroz))
maxx = max(points(:,1));maxy = max(points(:,2));
minx = min(points(:,1));miny = min(points(:,2));
maxz = high_point(:,3);
midx = (maxx + minx) / 2.0;
midy = (maxy + miny) / 2.0;
% ����ͶӰ��
zeroz = (b(1) + b(2)*b(2)*maxz + b(2)*midx + b(3)*b(3)*maxz + b(3)*midy)/(b(2)*b(2) + b(3)*b(3)+ (-1)*(-1));
zerox = (maxz - zeroz)*b(2) + midx;
zeroy = (maxz - zeroz)*b(3) + midy;
midz = (maxz + zeroz) / 2.0;
stone_r = norm([(midx-zerox),(midy-zeroy),(maxz-zeroz)]) / 2.0;

% ����
[x_,y_,z_]=sphere();
mesh(stone_r*x_+midx,stone_r*y_+midy,stone_r*z_+midz);
axis equal
hold off


%%%--------------------------------------------------------------------%%%%
% �������ϵĵ�ͶӰ��ԭͼ���д���
% Q_param = load('Q.txt');
% if size(Q_param,2) ~= 4
%     fprintf('Q file error\n');
% end
% Q = [1 0 0 Q_param(1);
%      0 1 0 Q_param(2);
%      0 0 0 Q_param(3);
%      0 0 Q_param(4) 0];
% ground(:,1:3) = ground(:,1:3).*ground(:,4);
% % img_data = inv(Q)*ground;
% img_data = Q\ground;
% img = zeros(600,800);
% for id = 1:size(img_data,1)
%     img(img_data(id,1),img_data(id,2)) = 1;
% end
figure(2);
subplot(2,2,1),imshow(img),title('ori');
imshow(img);

%%%--------------------------------------------------------------------%%%%
% ���д�����ȡʯͷ����
% ���ÿ�������ȥ���
se = strel('disk',8);%�ṹԪ��se
openimg=imopen(img,se);
subplot(2,2,2),imshow(openimg),title('open');
% ��ȡ��ͨ���򣬲�������ʾ
% [L,NUM] = bwlabel(openimg,8);
% RGB = label2rgb(L);
% subplot(2,2,3),imshow(RGB),title('rgb');
start_point = points(1,:);
start_point(1:3) = start_point(1:3)*start_point(4);
start_data = Q\start_point';
start_x = round(start_data(1));
start_y = round(start_data(2));
J = My_RegionGrowing(openimg, [start_x, start_y], 0);
subplot(2,2,3),imshow(J),title('grow');







