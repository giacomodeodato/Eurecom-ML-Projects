%Alessandro Patti
%Giacomo Deodato
close all
clear all
%%%%%%%%%%%%%%%%Pre-processing%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%Step 1%%%%%%%
img=imread('ic2.tif');
noise=randn(size(img,1),size(img,2));
noise_strength=64;
img_n=img+uint8(noise_strength*noise);

%%%%%%Step 2%%%%%%%

%calculate filtered image with 3 different methods and plot them to compare
%results. The size of the filter has been chosen after some test. The best
%result has been achieved with wiener filter, that preserver better the
%edges
mask_size=5;
img_fw=wiener2(img_n, [mask_size mask_size]);
img_fm=medfilt2(img_n, [mask_size mask_size]);
img_fa=imfilter(img_n,fspecial('average',mask_size));
subplot(2,2,1);
imshow(img_n)
title('original')
subplot(2,2,2);
imshow(img_fw)
title('wiener')
subplot(2,2,3);
imshow(img_fm)
title('median')
subplot(2,2,4);
imshow(img_fa)
title('average')


%%%%%%%%%%%%%%%%Processing%%%%%%%%%%%%%%%%%%%%%%%
%close all
figure
%%%%%%Step 3%%%%%%%
img_f=img_fw;

%%%%gradient edges detection%%%%
Gx=zeros(3,3);
Gx(2,1)=1; Gx(2,3)=-1; %create x-axis gradient mask
Gy=zeros(3,3);
Gy(1,2)=1; Gy(3,2)=-1; %create y-axis gradient mask
gradientx=imfilter(img_f,Gx); %calculate the gradient over x-axis direction
gradienty=imfilter(img_f,Gy); %calculate the gradient over y-axis direction

gradient=uint8(sqrt(double(gradientx.^2+gradienty.^2)));
% subplot(1,3,1);
% imshow(gradient,[]);
% title('gradient')
gradient=im2bw(gradient, graythresh(gradient));
% subplot(1,3,2);
% imshow(gradient,[]);
% title('gradient binarized ')
gradient=bwmorph(gradient,'thin');
% subplot(1,3,3);
% imshow(gradient,[]);
% title('gradient thin')
% figure

%%%%laplacian edges detection%%%
laplacian=imfilter(img_f,fspecial('laplacian'));
% subplot(1,3,1);
% imshow(laplacian);
% title('laplacian')
laplacian=im2bw(laplacian,graythresh(laplacian));
% subplot(1,3,2);
% imshow(laplacian);
% title('laplacian binarized')
laplacian=bwmorph(laplacian,'thin');
% subplot(1,3,3);
% imshow(laplacian);
% title('laplacian thin')
% figure

%%%%canny edges detection%%%%
canny=edge(img_f, 'canny', 0.4);

%%plot all the results
subplot(2,2,1);
imshow(img_f)
title('filtered')
subplot(2,2,2);
imshow(gradient)
title('gradient')
subplot(2,2,3);
imshow(laplacian)
title('laplacian')
subplot(2,2,4);
imshow(canny)
title('canny');

%%%%%%Step 4%%%%%%%
edges=canny;

%%%%show the radon and hough transform of a point and a line%%%
% figure
% point=zeros(256,256);
% point(50,50)=255;
% line=zeros(256,256);
% line(256/2+1,:)=255;
% subplot(1,2,1);
% imshow(radon(point),[])
% title('Radon: point')
% subplot(1,2,2);
% imshow(hough(point),[])
% title('Hough: point')
% figure
% subplot(1,2,1);
% imshow(log(radon(line)),[])
% title('Radon: log line')
% subplot(1,2,2);
% imshow(log(hough(line)),[])
% title('Hough: log line')

rad=radon(edges);
figure
imshow(rad, [])
title('radon transform')
%interactiveLine(edges, rad, 5);

high=zeros(1,180);
for i=1:180
    high(i)=max(rad(:,i));
end

maxsum=max(high(1:90)+high(91:180));
indexsum=find(high(1:90)+high(91:180)==maxsum);

figure
subplot(2,1,1)
plot(high)
title('full vector');
subplot(2,1,2)
plot(high(1:90)+high(91:180))
title('sum of orthogonal components');

figure
subplot(1,2,1)
imshow(img_n)
title('original noisy image');
subplot(1,2,2)
img_rot=imrotate(img_f,90-indexsum,'bicubic');
imshow(img_rot)
title('filtered image rotated');
