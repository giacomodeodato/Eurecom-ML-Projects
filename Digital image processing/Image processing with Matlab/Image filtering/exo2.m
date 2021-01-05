M = 64 * ones(256, 256);
M((256-64)/2+1:(256-64)/2+64,(256-64)/2+1:(256-64)/2+64) = 192;
noise = rand(256);

nstr = 10;
Mnoise = M + nstr * noise;
filter = fspecial('average', 3);
Mfiltered = filter2(filter, Mnoise);
figure;
subplot(3, 3, 1);
imshow(uint8(M));
title('original');
subplot(3, 3, 2);
imshow(uint8(Mnoise));
title('noise strength: 10');
subplot(3, 3, 3);
imshow(uint8(Mfiltered));
title('filter size: 3');

nstr = 10;
Mnoise = M + nstr * noise;
filter = fspecial('average', 9);
Mfiltered = filter2(filter, Mnoise);
subplot(3, 3, 4);
imshow(uint8(M));
title('original');
subplot(3, 3, 5);
imshow(uint8(Mnoise));
title('noise strength: 10');
subplot(3, 3, 6);
imshow(uint8(Mfiltered));
title('filter size: 9');

nstr = 50;
Mnoise = M + nstr * noise;
filter = fspecial('average', 9);
Mfiltered = filter2(filter, Mnoise);
subplot(3, 3, 7);
imshow(uint8(M));
title('original');
subplot(3, 3, 8);
imshow(uint8(Mnoise));
title('noise strength: 50');
subplot(3, 3, 9);
imshow(uint8(Mfiltered));
title('filter size: 9');