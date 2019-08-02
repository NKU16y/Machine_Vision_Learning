function Moravec_myself=Moravec_myself(img,r,Tradio)
[m,n,d]=size(img);
%��ȡ���ݴ�С
%ת��Ϊ�Ҷ�ͼ
grayimg=rgb2gray(img);
w=ones(r);
R=zeros(m,n);
Size=floor(r/   2);
Rmax=0;
%��������غ���R�����ں���Ϊr*r��ȫ1����
for i=Size+1:m-Size-1
    for j=Size+1:n-Size-1
        V1=0;V2=0;V3=0;V4=0;
        for k=-Size:Size
            for h=-Size:Size
            V1=V1+(double(grayimg(i+k,j+h))-double(grayimg(i+k+1,j+h))).^2;%��ֱ����
            V2=V2+(double(grayimg(i+k,j+h))-double(grayimg(i+k,j+h+1))).^2;%ˮƽ����
            V3=V3+(double(grayimg(i+k,j+h))-double(grayimg(i+k+1,j+h+1))).^2;%����
            if j+h>1% ��ֹ�����±�Ϊ0
            V4=V4+(double(grayimg(i+k,j+h))-double(grayimg(i+k+1,j+h-1))).^2;%����
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
figure('NumberTitle','off','name','Moravec_result');
imshow(img);
hold on
plot(r,c,'r*');
                 