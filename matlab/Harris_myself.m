function Harris_myself=Harris_myself(img,sigma,Tradio,k,r)
%参数：高斯函数参数σ,阈值占最大值比重Tradio,参数k,窗口大小r
%转化为灰度图
grayimg=rgb2gray(img);
grayimg=double(grayimg);
[m,n]=size(grayimg);
%计算自相关矩阵M
%计算Ix,Iy
Ix=zeros(m,n);  
Iy=zeros(m,n); 
%用sobel算子作差分，近似当做Ix,Iy
hx=fspecial('sobel');%水平梯度模板
hy=hx';%垂直梯度模板
Ix=imfilter(grayimg,hx,'conv', 'replicate'); %计算Ix
Iy=imfilter(grayimg,hy,'conv', 'replicate'); %计算Iy
Ix2=Ix.^2;
Iy2=Iy.^2;%分别求平方
Ixy=Ix.*Iy;
H=fspecial('gaussian',r , sigma);%产生高斯窗口函数
Size=floor(r/2);
R=zeros(m,n);
Rmax=0;
%计算角点响应函数R，同时进行阈值检测
for i=Size+1:m-Size-1
    for j=Size+1:n-Size-1
        %用高斯函数加权
        a=Ix2(i-Size:i+Size,j-Size:j+Size);
        b=Iy2(i-Size:i+Size,j-Size:j+Size);
        c=Ixy(i-Size:i+Size,j-Size:j+Size);
        A1=filter2(H,a);
        B1=filter2(H,b);
        C1=filter2(H,c);
        A2=sum(A1);
        B2=sum(B1);
        C2=sum(C1);
        A=sum(A2);
        B=sum(B2);
        C=sum(C2);
        M=[A,C;C,B];  
        R(i,j)=det(M)-k*(trace(M))^2;
        if R(i,j)>Rmax
        Rmax=R(i,j);
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
figure('NumberTitle','off','name','Harris_result');
imshow(img);
hold on
plot(r,c,'r+');