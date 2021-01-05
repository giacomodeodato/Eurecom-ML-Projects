close all

% Read left and right images
imgR = imread('Dist=2/x=0_1.jpg');
imgL = imread('Dist=2/x=-0_1.jpg');

% Resize source images because stereo matching is a time consuming process
scale_factor = 0.4;
imgL = imresize(imgL, scale_factor);
imgR = imresize(imgR, scale_factor);

% Apply the matching process for the stereo pair using the Matching()
% function.
% For the matching process try different different block sizes (3 and 9).
dmax = 45;
block_size = [3, 9];
disp_map3 = Matching(imgL, imgR, dmax, block_size(1));
disp_map9 = Matching(imgL, imgR, dmax, block_size(2));

% Save the disparity under the format of image
imwrite(uint8(disp_map3), 'disp_map2_01_3.png');
imwrite(uint8(disp_map9), 'disp_map2_01_9.png');

% Display the two disparity maps
figure;
imshow(imread('disp_map2_01_3.png'), []);
title('disp map: dist=2; x=+-0.1; block size=3;');
figure;
imshow(imread('disp_map2_01_9.png'), []);
title('disp map: dist=2; x=+-0.1; block size=9;');

% Using small blocks you get noisy resultimg disparity map.
% Apply average filtering to remove noise which possibly belongs to
% incorrect matching.
figure;
disp_map3_filtered = imfilter(disp_map3, fspecial('average', 3));
imshow(disp_map3_filtered, []);
title('avg filter disp map: dist=2; x=+-0.4; block size=3;');

% Apply the matching process for the other 3 stereo pairs.
figure;
subplot(2, 3, 1);
imshow(imread('disp_map2_01_3.png'), []);
title('disp map: dist=2; x=+-0.1; block size=3;');
subplot(2, 3, 2);
imshow(imread('disp_map2_01_9.png'), []);
title('disp map: dist=2; x=+-0.1; block size=9;');

%%% Dist = 2; X = +-0.4
block_size = 1;
imgR = imread('Dist=2/x=0_4.jpg');
imgL = imread('Dist=2/x=-0_4.jpg');
imgL = imresize(imgL, scale_factor);
imgR = imresize(imgR, scale_factor);
dmax = 170;
disp_map = Matching(imgL, imgR, dmax, block_size);
imwrite(uint8(disp_map), 'disp_map2_04_1.png');
subplot(2, 3, 3);
imshow(imread('disp_map2_04_1.png'), []);
title('disp map: dist=2; x=+-0.4; block size=1;');

%%% Dist = 4; X = +-0.1
block_size = 1;
imgR = imread('Dist=4/x=0_1.jpg');
imgL = imread('Dist=4/x=-0_1.jpg');
imgL = imresize(imgL, scale_factor);
imgR = imresize(imgR, scale_factor);
dmax = 22;
disp_map = Matching(imgL, imgR, dmax, block_size);
imwrite(uint8(disp_map), 'disp_map4_01_1.png');
subplot(2, 3, 4);
imshow(imread('disp_map4_01_1.png'), []);
title('disp map: dist=4; x=+-0.1; block size=1;');

%%% Dist = 4; X = +-0.4
block_size = 1;
imgR = imread('Dist=4/x=0_4.jpg');
imgL = imread('Dist=4/x=-0_4.jpg');
imgL = imresize(imgL, scale_factor);
imgR = imresize(imgR, scale_factor);
dmax = 85;
disp_map = Matching(imgL, imgR, dmax, block_size);
imwrite(uint8(disp_map), 'disp_map4_04_1.png');
subplot(2, 3, 5);
imshow(imread('disp_map4_04_1.png'), []);
title('disp map: dist=4; x=+-0.4; block size=1;');