clc; clear; close all;

%% ===== 0) Load image =====
imgPath = 'original_img.JPG'; 
img = imread(imgPath);

if ndims(img) == 2
    img = repmat(img,1,1,3);
end

[rows, cols, ch] = size(img);
N = rows*cols;

%% ===== 1) Original Image =====
figure('Name','Original Image');
imagesc(img); axis image off;
title('Original Image');

%% ===== 1) Histogram of Original Image (2x2) =====
figure('Name','Histogram of Original Image');

subplot(2,2,1);
imagesc(img); axis image off;
title('Original Image');

subplot(2,2,2);
plotHistColor(img(:,:,1), [1 0 0]); title('Histogram - Red');
xlabel('Pixel Intensity'); ylabel('Frequency');

subplot(2,2,3);
plotHistColor(img(:,:,2), [0 1 0]); title('Histogram - Green');
xlabel('Pixel Intensity'); ylabel('Frequency');

subplot(2,2,4);
plotHistColor(img(:,:,3), [0 0 1]); title('Histogram - Blue');
xlabel('Pixel Intensity'); ylabel('Frequency');

%% ===== 2) Confusion Stage (Scrambling Logistic Map) =====
r = 3.99;
x0_scramble = 0.7; 

img1d = reshape(img, [], ch);
seq = zeros(N,1);

x = x0_scramble;
for i=1:N
    x = r*x*(1-x);
    seq(i) = x;
end

[~, idx] = sort(seq);
scrambled = img1d(idx,:);
scrambled_img = reshape(scrambled, rows, cols, ch);

figure('Name','Scrambled Image - Confusion Stage');
imagesc(scrambled_img); axis image off;
title('Scrambled Image (Confusion)');

%% ===== 3) Encrypted Image - DNA Diffusion Stage (XOR on bits) =====
x0_diffusion = 0.5;  
x = x0_diffusion;

flatVals = scrambled(:);         
binChars = dec2bin(flatVals, 8);  
binMat   = binChars - '0';        

totalBits = numel(binMat);
chaosBits = zeros(totalBits,1,'uint8');

for i=1:totalBits
    x = r*x*(1-x);
    chaosBits(i) = uint8(mod(floor(x*1e14),2));
end

flatBin = reshape(binMat.', [], 1);
xored   = xor(uint8(flatBin), chaosBits);

xored8  = reshape(char(xored+'0'), 8, []).';
decVals = uint8(bin2dec(xored8));

encrypted_img = reshape(decVals, rows, cols, ch);

figure('Name','Encrypted Image - DNA Diffusion Stage');
imagesc(encrypted_img); axis image off;
title('Encrypted Image (DNA Diffusion)');

%% ===== 5) Histogram After Encryption (RGB) =====
figure('Name','Histogram After Encryption');

subplot(1,3,1);
plotHistColor(encrypted_img(:,:,1), [1 0 0]); title('Encrypted Histogram - Red');
xlabel('Pixel Intensity'); ylabel('Frequency');

subplot(1,3,2);
plotHistColor(encrypted_img(:,:,2), [0 1 0]); title('Encrypted Histogram - Green');
xlabel('Pixel Intensity'); ylabel('Frequency');

subplot(1,3,3);
plotHistColor(encrypted_img(:,:,3), [0 0 1]); title('Encrypted Histogram - Blue');
xlabel('Pixel Intensity'); ylabel('Frequency');

%% ===== 6) Side-by-Side Comparison =====
figure('Name','Side-by-Side Comparison');
subplot(1,3,1); imagesc(img); axis image off; title('Original Image');
subplot(1,3,2); imagesc(scrambled_img); axis image off; title('Scrambled Image (Confusion)');
subplot(1,3,3); imagesc(encrypted_img); axis image off; title('Encrypted Image (DNA Diffusion)');

%% ===== 7) Entropy Analysis (Per Channel) =====
fprintf('\n7. Entropy Analysis (Per Channel)\n');

H_R_orig = entropy_manual(img(:,:,1));
H_G_orig = entropy_manual(img(:,:,2));
H_B_orig = entropy_manual(img(:,:,3));

H_R_enc  = entropy_manual(encrypted_img(:,:,1));
H_G_enc  = entropy_manual(encrypted_img(:,:,2));
H_B_enc  = entropy_manual(encrypted_img(:,:,3));

fprintf('Original Entropy  (R,G,B) = %.5f  %.5f  %.5f\n', H_R_orig, H_G_orig, H_B_orig);
fprintf('Encrypted Entropy (R,G,B) = %.5f  %.5f  %.5f\n', H_R_enc,  H_G_enc,  H_B_enc);

%% ===== 8) Correlation Analysis (Per Channel - Horizontal Adjacent) =====
fprintf('\n8. Correlation Analysis (Per Channel - Horizontal Adjacent)\n');

cR_orig = adjacent_corr_horizontal(img(:,:,1));
cG_orig = adjacent_corr_horizontal(img(:,:,2));
cB_orig = adjacent_corr_horizontal(img(:,:,3));

cR_enc  = adjacent_corr_horizontal(encrypted_img(:,:,1));
cG_enc  = adjacent_corr_horizontal(encrypted_img(:,:,2));
cB_enc  = adjacent_corr_horizontal(encrypted_img(:,:,3));

fprintf('Original Correlation  (R,G,B) = %.4f  %.4f  %.4f\n', cR_orig, cG_orig, cB_orig);
fprintf('Encrypted Correlation (R,G,B) = %.4f  %.4f  %.4f\n', cR_enc,  cG_enc,  cB_enc);


figure('Name','Correlation Analysis - Per Channel');
rng(1); 

subplot(2,3,1); scatter_adjacent(img(:,:,1), 8000); title(sprintf('Original R (r=%.4f)', cR_orig)); xlabel('Pixel(i)'); ylabel('Pixel(i+1)'); grid on;
subplot(2,3,2); scatter_adjacent(img(:,:,2), 8000); title(sprintf('Original G (r=%.4f)', cG_orig)); xlabel('Pixel(i)'); ylabel('Pixel(i+1)'); grid on;
subplot(2,3,3); scatter_adjacent(img(:,:,3), 8000); title(sprintf('Original B (r=%.4f)', cB_orig)); xlabel('Pixel(i)'); ylabel('Pixel(i+1)'); grid on;

subplot(2,3,4); scatter_adjacent(encrypted_img(:,:,1), 8000); title(sprintf('Encrypted R (r=%.4f)', cR_enc)); xlabel('Pixel(i)'); ylabel('Pixel(i+1)'); grid on;
subplot(2,3,5); scatter_adjacent(encrypted_img(:,:,2), 8000); title(sprintf('Encrypted G (r=%.4f)', cG_enc)); xlabel('Pixel(i)'); ylabel('Pixel(i+1)'); grid on;
subplot(2,3,6); scatter_adjacent(encrypted_img(:,:,3), 8000); title(sprintf('Encrypted B (r=%.4f)', cB_enc)); xlabel('Pixel(i)'); ylabel('Pixel(i+1)'); grid on;

%% ===== Helper functions (NO toolbox) =====
function H = entropy_manual(channel_uint8)
    counts = histcounts(channel_uint8(:), 0:256);
    p = counts / sum(counts);
    p(p==0) = [];
    H = -sum(p .* log2(p));
end

function r = adjacent_corr_horizontal(channel_uint8)
    A = double(channel_uint8);
    x = A(:,1:end-1); y = A(:,2:end);
    x = x(:); y = y(:);
    x = x - mean(x); y = y - mean(y);
    denom = sqrt(sum(x.^2) * sum(y.^2));
    if denom==0, r=0; else, r = sum(x.*y)/denom; end
end

function scatter_adjacent(channel_uint8, N)
    A = double(channel_uint8);
    x = A(:,1:end-1); y = A(:,2:end);
    x = x(:); y = y(:);
    M = numel(x);
    if N > M, N = M; end
    idx = randperm(M, N);
    plot(x(idx), y(idx), '.', 'MarkerSize', 4);
end

function plotHistColor(channel_uint8, rgb)
    counts = histcounts(channel_uint8(:), 0:256);
    b = bar(0:255, counts);
    b.FaceColor = rgb;
    b.EdgeColor = rgb;
    xlim([0 255]);
end
