close all

% read source images
img0 = imread('Pair_anaglyph/x=0.0.jpg');
img1 = imread('Pair_anaglyph/x=0.1.jpg');

% copy the red color channel of the left image
img(:,:,1)=img0(:,:,1);

% copy the green and blue channels of the right image
img(:,:,2)=img1(:,:,2);
img(:,:,3)=img1(:,:,3);

% show the result
figure;
imshow(img);
title('Flower Anaglyph');

% show the lion statue anaglyph
figure;
imshow(imread('Lion_anaglyph/Lion_Statue_anaglyph.jpg'));
title('Lion statue anaglyph');