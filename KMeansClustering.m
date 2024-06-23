function KMeansClustering

%Task a: Reading the Input image and setting up a 3D RGB feature space
%---------------------------------------------------------------------------
%Read the Input Image
inputImage = imread("inputEx5_1.jpg");

%Convert the Image to doouble for more precision in calculations
inputImage = im2double(inputImage);

%Obtain the matrix dimensions of the image
[rows, cols, channels] = size(inputImage);

%Reshape the image into a Nx3 matrix where N = row * col
RGBfeatureSpace = reshape(inputImage, rows * cols, channels);

%Display the input image
figure; imshow(inputImage);
title("Input Image");

%Display the scatter plot of the RGB feature space
figure;
scatter3(RGBfeatureSpace(:, 1), RGBfeatureSpace(:, 2), RGBfeatureSpace(:, 3), 1, RGBfeatureSpace, 'filled');
title("3D RGB Feature Space");
xlabel('Red Channel');
ylabel('Green Channel');
zlabel('Blue Channel');
grid on;

%Task c: Implement a random Cluster with the colour feature, update and
%visualize
%---------------------------------------------------------------
k = 8; % Number of cluster assumed
maxIter = 100; %Maximum number of iterations

%Calculate the cluster points and centriods
[clusterPoints, centriods] = CalculateKMeans(RGBfeatureSpace, k, maxIter);

%Reshape cluster points back to original image dimensions
clusteredImage = reshape(clusterPoints, rows, cols);

%Visulaizing the clustered image
figure; imagesc(clusteredImage);
title(['Clustered Image with ', num2str(k), ' Clusters']);
colormap(jet);
colorbar;

%Task d. Extend the 3D feature space with additional spatial support using
%pixel positions (x, y) and test algorithm for the 5D space
%---------------------------------------------------------------------------
%create the spatial coordinates
[X, Y] = meshgrid(1:cols, 1:rows);
spatialCoordinates = [X(:), Y(:)];

%Combining the RGB feastures with spatial coordinates to create 5D feature
%space
featurespace5D = [RGBfeatureSpace, spatialCoordinates];

%calculate the cluster points and centriod of the 5D feature space
[clusterpoints5D, centriods5D] = CalculateKMeans(featurespace5D, k, maxIter);

%Reshape the cluster points  back to original image dimensions
clusteredImage5D = reshape(clusterpoints5D, rows, cols);

%Visualizing the clustered image for the 5D feature space
figure; imagesc(clusteredImage5D);
title(['Clustered Image with ', num2str(k), ' Clusters (5D Feature Space)']);
colormap(jet);
colorbar;

disp("The suitable k value that displays all shapes and features properly is k value = ", num2str(k));
disp("The 5D space segments the shape better however, the geometry of the input image is unrecognisable");

end

%Implementing a function to help calculate Kmeans
function [points, centriods] = CalculateKMeans(featureMatrix, k, maxIter)

%Task b: Randomly initialize the centriods
centriods = featureMatrix(randsample(size(featureMatrix, 1), k), :);

%Initializing cluster points
points = zeros(size(featureMatrix, 1), 1);
prevCentriods = centriods;

for iter = 1 : maxIter
    %Assigning each points to the nearest centriod
    for i = 1 : size(featureMatrix, 1)
        distances = sum((featureMatrix(i, :) - centriods) .^2, 2);
        [~, points(i)] = min(distances);
    end

    %Update the centriod
    for j = 1 : k
        clusterPoints = featureMatrix(points == j, :);
        if ~isempty(clusterPoints)
            centriods(j, :) = mean(clusterPoints, 1);
        end
    end

    %Checking for Convergence
    if isequal(prevCentriods, centriods)
        break;
    end

    %updating previous centriods
    prevCentriods = centriods;
end
end