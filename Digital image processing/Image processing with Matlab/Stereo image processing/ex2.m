close all

% Read the source file and convert the disparity map to depth map.
disp_map = imread('ground.png');
depth_map = DepthCompute(disp_map, 30, 600);

figure;
subplot(1, 3, 1);
imshow(disp_map, []);
title("Disparity Map");
subplot(1, 3, 2);
imshow(depth_map, []);
title("Depth Map");

% Apply median filtering to get a smoother map.
depth_map_filtered = medfilt2(depth_map, [19 19]);
subplot(1, 3, 3);
imshow(depth_map_filtered, []);
title("Depth Map filtered. Median filter size: 19");

% Load the left image of the stereo pair.
imgL = imread('view1.png');

% For 3D reconstruction warp the left image of the stereo pair on depth
% data.
% Show the 3D reconstruction from different viewpoints.
figure;
warp(-depth_map_filtered, imgL);
title("Warp of the image on the stereo data. Hit ENTER to change view.");
rotate3d on
view([0,90]);
pause;
view([0,0]);
pause;
view([60,60]);
pause;

% Show distribution of the values.
figure;
imhist(uint8(depth_map_filtered));
title('Distribution of grey values in the depth map')

% Subtract the background
x = ginput(1);
background = find(depth_map_filtered > x(1));
depth_map_filtered(background) = 255;
depth_map_filtered = 255-depth_map_filtered;
figure;
imshow(uint8(depth_map_filtered),[]);
title("Background subtraction");

% Separate intermediate layers and foreground
figure;
imhist(uint8(depth_map_filtered));
title('Identify to thesholds for intermediate planes');
x = ginput(2);
if x(1) > x(2)
    tmp = x(1);
    x(1) = x(2);
    x(2) = tmp;
end
[N, M] = size(depth_map_filtered);
rgb_map = zeros([N M 3]);

[row, col] = find(depth_map_filtered < x(1) & depth_map_filtered > 0);
for i = 1:size(row)
    rgb_map(row(i), col(i), 1) = 255;
end
[row, col] = find(depth_map_filtered < x(2) & depth_map_filtered > x(1));
for i = 1:size(row)
    rgb_map(row(i), col(i), 2) = 255;
end
[row, col] = find(depth_map_filtered > x(2));
for i = 1:size(row)
    rgb_map(row(i), col(i), 3) = 255;
end
imshow(rgb_map);