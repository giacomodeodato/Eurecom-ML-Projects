function [ DepthMap] = DepthCompute (disp, B, f )
    [N, M] = size(disp);
    DepthMap = zeros(N,M);
    max = 255;
    
    for i = 1:1:N
        for j = 1:1:M
            if (disp(i,j) == 0)
                DepthMap(i, j) = max;
            else
                DepthMap(i,j) = f * B / disp(i, j);
            end
        end
    end
