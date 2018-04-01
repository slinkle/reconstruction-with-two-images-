function J = My_RegionGrowing(I, init_pos, reg_maxdist)
% ������������ȡĿ�����򣺱Ƚ���������������ƽ���Ҷ�ֵ����������صĻҶ�ֵ
% ���룺
%    I : ��ά���飬��ֵ��ʾ�Ҷ�ֵ��0~255
%    init_pos: ָ�������ӵ�����
%    reg_maxdist : ��ֵ��Ĭ��ֵΪ20
% �����
%   J : ����Ȥ����
% ʾ����
%{
img = imread('1.bmp');
I = rgb2gray(img); 
I = double(I);
x = 271;
y = 259;
J = My_RegionGrowing(I, [x, y], 20);
imshow(img)
hold on 
plot(y, x, 'p')
hold off
figure
imagesc(J)
%}
[row, col] = size(I);               % ����ͼ���ά�� 
J = zeros(row, col);               % ���
x0 = init_pos(1);                   % ��ʼ��
y0 = init_pos(2);
reg_mean = I(x0, y0);       % ������ʼ��Ҷ�ֵ
J(x0, y0) = 1;                    % ������ʼ������Ϊ��ɫ
reg_sum = reg_mean;                     % �������������ĻҶ�ֵ�ܺ�
reg_num = 1;                                  % �������������ĵ�ĸ���
count = 1;                                       % ÿ���ж���Χ�˸����з�����������Ŀ
reg_choose = zeros(row*col, 2);      % ��¼��ѡ��������
reg_choose(reg_num, :) = init_pos;
num = 1;               % ��һ����   
while count > 0
    s_temp = 0;                  % ��Χ�˸����з��������ĵ�ĻҶ�ֵ�ܺ�
    count = 0;
    for k = 1 : num      % ��������ÿ��������������ظ�
        i = reg_choose(reg_num - num + k, 1);
        j = reg_choose(reg_num - num +k, 2);
        if J(i, j) == 1 && i > 1 && i < row && j > 1 && j < col   % ��ȷ���Ҳ��Ǳ߽��ϵĵ�
            % ������
            for u =  -1 : 1      
                for v = -1 : 1
                    % δ�������������������ĵ�
%                     �˴�reg_maxdist = 0
                    if J(i + u, j + v) == 0 && abs(I(i + u, j + v) - reg_mean) <= reg_maxdist
                        J(i + u, j + v) = 1;           % ��Ӧ������Ϊ��ɫ
                        count = count + 1;
                        reg_choose(reg_num + count, :) = [i + u, j + v];
                        s_temp = s_temp + I(i + u, j + v);   % �Ҷ�ֵ����s_temp��
                    end
                end
            end
        end
    end
    num = count;                                      % �����ĵ�
    reg_num = reg_num + count;              % �������ܵ���
    reg_sum = reg_sum + s_temp;            % �������ܻҶ�ֵ
    reg_mean = reg_sum / reg_num;         % ����Ҷ�ƽ��ֵ
end