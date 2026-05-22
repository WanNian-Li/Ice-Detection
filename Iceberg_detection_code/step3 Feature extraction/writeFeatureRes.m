function writeFeatureRes(filename1, filename2, filename)
tic
% Read the label and image data from the GeoTIFF files
[L_all, ~] = geotiffread(filename1);
[im_all, ~] = geotiffread(filename2);

% Normalize image values (uncomment if needed)
% im_all = double(im_all) ./ 1000;

% Replace NaNs with a fill value
im_all(isnan(im_all)) = -9999;

% im_all = db2mag(im_all);  % Uncomment if needed

subsize = 2000;
% Get image dimensions
numRows = size(im_all, 1);
numCols = size(im_all, 2);
sub_nRow = floor(numRows / subsize) + 1;
sub_nCol = floor(numCols / subsize) + 1;

% Calculate boundary indices for rows and columns
Row_boundary = (0:sub_nRow) * subsize + 1;
Row_boundary(end) = numRows;
Col_boundary = (0:sub_nCol) * subsize + 1;
Col_boundary(end) = numCols;
% Adjust the last boundary to ensure valid indexing
Row_boundary(end) = Row_boundary(end) + 1;
Col_boundary(end) = Col_boundary(end) + 1;

N = 0;
features_all = zeros(10e6, 27);
% Process the image in sub-blocks
for i = 1:sub_nRow
    for j = 1:sub_nCol
        im = im_all(Row_boundary(i):Row_boundary(i+1)-1, Col_boundary(j):Col_boundary(j+1)-1);
        L = L_all(Row_boundary(i):Row_boundary(i+1)-1, Col_boundary(j):Col_boundary(j+1)-1);
        [superpx_test, n] = sample_label2(im, L);
        if n > 0
            features_all(N+1:N+n, :) = superpx_test;
            features_all(N+1:N+n, 2) = (N+1:N+n)';
            clear superpx_test;
            N = N + n;
        end
    end
end
% Remove unused preallocated rows
features_all(N+1:10e6, :) = [];

%% Save features_all to a .nc file using NetCDF functions
% Create a new NetCDF file (overwrite if it exists)
ncid = netcdf.create(filename, 'CLOBBER');

% Define two dimensions: the number of samples and the number of features
dimid_sample = netcdf.defDim(ncid, 'sample', N);
dimid_feature = netcdf.defDim(ncid, 'feature', 27);

% Define the variable 'features_all' with dimensions [feature, sample]
varid = netcdf.defVar(ncid, 'features_all', 'double', [dimid_feature, dimid_sample]);

% End define mode
netcdf.endDef(ncid);

% Write the data to the NetCDF file (transpose to match dimension order)
netcdf.putVar(ncid, varid, features_all');

% Close the NetCDF file
netcdf.close(ncid);

%% Finished saving
toc
end
