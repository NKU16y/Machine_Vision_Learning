function Moravec_myself=Moravec_myself(img,r,Tradio)
[m,n,d]=size(img);
%获取数据大小
%转化为灰度图
grayimg=rgb2gray(img);
w=ones(r);
R=zeros(m,n);
Size=floor(r/   2);
Rmax=0;
%计算自相关函数R，窗口函数为r*r的全1矩阵
for i=Size+1:m-Size-1
    for j=Size+1:n-Size-1
        V1=0;V2=0;V3=0;V4=0;
        for k=-Size:Size
            for h=-Size:Size
            V1=V1+(double(grayimg(i+k,j+h))-double(grayimg(i+k+1,j+h))).^2;%竖直方向
            V2=V2+(double(grayimg(i+k,j+h))-double(grayimg(i+k,j+h+1))).^2;%水平方向
            V3=V3+(double(grayimg(i+k,j+h))-double(grayimg(i+k+1,j+h+1))).^2;%右下
            if j+h>1% 防止出现下标为0
            V4=V4+(double(grayimg(i+k,j+h))-double(grayimg(i+k+1,j+h-1))).^2;%左下
            end
            R(i,j)=min([V1,V2,V3,V4]);
            if R(i,j)>Rmax
                Rmax=R(i,j);
            end
            end
        end
    end
end
T=Tradio*Rmax;
%阈值检测
R(R<T)=0;
%非极大值抑制
for i=Size+1:m-Size-1
    for j=Size+1:n-Size-1
       Max1=max(R(i-Size:i+Size,j-Size:j+Size));
       Max=max(Max1);
       if R(i,j)<Max
          R(i,j)=0;
       end
    end
end
%选择非零点作为角点检测结果
c=[];
r=[];
for i=Size+1:m-Size-1
    for j=Size+1:n-Size-1
        if R(i,j)~=0
            r=[r;j];
            c=[c;i];
        end
    end
end
%画图
figure('NumberTitle','off','name','Moravec_result');
imshow(img);
hold on
plot(r,c,'r*');
                 