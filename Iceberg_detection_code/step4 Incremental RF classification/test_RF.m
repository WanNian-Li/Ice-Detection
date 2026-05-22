clc;
clear;
% Run radom forest classifier to extract icebergs.
% Directory for feature (.nc) and label (.tif) files
datadir1 = 'features_202110/';
filelist1 = dir([datadir1, '*.nc']);
datadir2 = 'Label_202110/';
filelist2 = dir([datadir2, '*.tif']);

% Extract grid numbers from feature file names
for i = 1:length(filelist1)
    name = strsplit(filelist1(i).name, '_');
    temp = name{2};  % use {} for cell indexing
    file_grid_f(i) = str2num(temp);
end

% Extract grid numbers from label file names
for i = 1:length(filelist2)
    name = strsplit(filelist2(i).name, '_');
    temp = name{2};
    file_grid_l(i) = str2num(temp);
end

% Get training data from 'trian_data/' folder (assumes at least one .nc file exists)
train_dir = 'trian_data/';
train_list = dir(fullfile(train_dir, '*.nc'));
if isempty(train_list)
    error('No training data .nc file found in trian_data folder.');
end
filename_train = fullfile(train_dir, train_list(1).name);

iteration = 30;
for p = 1:length(filelist1)
    tic;
    filename_feature = fullfile(datadir1, filelist1(p).name);
    pos = find(file_grid_l == file_grid_f(p));
    filename_label = fullfile(datadir2, filelist2(pos).name);
    shp_filename = fullfile('res202110_shp/', [filelist1(p).name(1:19), '_resb.shp']);
    
    [L_res, R] = classify_iceberg(filename_train, filename_feature, filename_label, iteration);
    writeVectorres(L_res, shp_filename, R);
    toc;
    disp(['done: ', filename_feature]);
end
