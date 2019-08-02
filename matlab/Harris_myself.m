function Harris_myself=Harris_myself(img,sigma,Tradio,k,r)
%��������˹����������,��ֵռ���ֵ����Tradio,����k,���ڴ�Сr
%ת��Ϊ�Ҷ�ͼ
grayimg=rgb2gray(img);
grayimg=double(grayimg);
[m,n]=size(grayimg);
%��������ؾ���M
%����Ix,Iy
Ix=zeros(m,n);  
Iy=zeros(m,n); 
%��sobel��������֣����Ƶ���Ix,Iy
hx=fspecial('sobel');%ˮƽ�ݶ�ģ��
hy=hx';%��ֱ�ݶ�ģ��
Ix=imfilter(grayimg,hx,'conv', 'replicate'); %����Ix
Iy=imfilter(grayimg,hy,'conv', 'replicate'); %����Iy
Ix2=Ix.^2;
Iy2=Iy.^2;%�ֱ���ƽ��
Ixy=Ix.*Iy;
H=fspecial('gaussian',r , sigma);%������˹���ں���
Size=floor(r/2);
R=zeros(m,n);
Rmax=0;
%����ǵ���Ӧ����R��ͬʱ������ֵ���
for i=Size+1:m-Size-1
    for j=Size+1:n-Size-1
        %�ø�˹������Ȩ
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
%��ֵ���
R(R<T)=0;
%�Ǽ���ֵ����
for i=Size+1:m-Size-1
    for j=Size+1:n-Size-1
       Max1=max(R(i-Size:i+Size,j-Size:j+Size));
       Max=max(Max1);
       if R(i,j)<Max
          R(i,j)=0;
       end
    end
end
%ѡ��������Ϊ�ǵ�����
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
%��ͼ
figure('NumberTitle','off','name','Harris_result');
imshow(img);
hold on
plot(r,c,'r+');