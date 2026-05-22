function [L_res, R] = classify_iceberg(filename_train, filename_feature, filename_label, iteration)
% Read training data from NetCDF file.
% The training file stores variable 'features_all' with dimensions [feature x sample],
% so transpose it to obtain a [sample x feature] matrix.
feature_train = ncread(filename_train, 'train_data')';
% Read test features from NetCDF file and transpose similarly
feature_test = ncread(filename_feature, 'features_all')';

% Read label image and its spatial reference from the GeoTIFF file
[L_all, R] = geotiffread(filename_label);

% Remove training samples with both feature columns 4 and 16 equal to 0
temp1 = feature_train(:,4);
temp2 = feature_train(:,16);
feature_train(temp1==0 & temp2==0, :) = [];
[m, n] = size(L_all);
L_res = zeros(m, n);
p = feature_train(:,3);
sample_count = length(find(p == 1));

% Define classifier weights and threshold
weights = [0.218, 0.271, 0.246, 0.265];
thres = 0.783;

for j = 1:iteration
    % Extract iceberg and non-iceberg features using selected columns
    feature_iceberg(:,1) = feature_train(p==1, 9);
    feature_iceberg(:,2) = feature_train(p==1, 7);
    feature_iceberg(:,3) = feature_train(p==1, 15);
    feature_noniceberg(:,1) = feature_train(p==0, 9);
    feature_noniceberg(:,2) = feature_train(p==0, 7);
    feature_noniceberg(:,3) = feature_train(p==0, 15);
    
    % Compute Mahalanobis distances
    D_i = mahal(feature_iceberg, feature_iceberg);
    D_non = mahal(feature_noniceberg, feature_iceberg);
    mean_d_i = mean(D_i);
    std_d_i = std(D_i);
    mean_d_non = mean(D_non);
    std_d_non = std(D_non);
    
    count(j) = length(feature_train);
    
    % Train Random Forest classifiers on different feature subsets
    TB1{j} = TreeBagger(200, feature_train(:,4:7), feature_train(:,3), 'Method', 'classification');
    TB2{j} = TreeBagger(100, feature_train(:,8:14), feature_train(:,3), 'Method', 'classification');
    TB3{j} = TreeBagger(250, feature_train(:,15:27), feature_train(:,3), 'Method', 'classification');
    TB5{j} = TreeBagger(150, feature_train(:,4:27), feature_train(:,3), 'Method', 'classification');
    
    % Predict using the trained classifiers
    P1 = predict(TB1{j}, feature_test(:,4:7));
    P2 = predict(TB2{j}, feature_test(:,8:14));
    P3 = predict(TB3{j}, feature_test(:,15:27));
    P5 = predict(TB5{j}, feature_test(:,4:27));
    P1_ = str2num(cell2mat(P1));
    P2_ = str2num(cell2mat(P2));
    P3_ = str2num(cell2mat(P3));
    P5_ = str2num(cell2mat(P5));
    
    % Stop iteration if the distance criterion is met
    if (mean_d_non - mean_d_i) < 2 * std_d_i
        break;
    end
    
    % Construct evaluation matrix: combine Mahalanobis distance and classifier scores
    feature_temp(:,1) = feature_test(:,9);
    feature_temp(:,2) = feature_test(:,7);
    feature_temp(:,3) = feature_test(:,15);
    D = mahal(feature_temp, feature_iceberg);
    marks = weights(1)*P1_ + weights(2)*P2_ + weights(3)*P3_ + weights(4)*P5_;
    marks(marks > thres) = 5.5;
    f_label = feature_test(:,3);
    D(D < std_d_i) = 1;
    D(D > (mean_d_non + std_d_non)) = 0;
    f_label(f_label == 2) = 100;
    f_score = D + marks + f_label;
    t_iceberg = find(f_score == 104);
    t_non_all = find(f_score == 100);
    
    % Balance iceberg and non-iceberg samples
    if length(t_non_all) > length(t_iceberg)
        random_num = t_non_all(randperm(numel(t_non_all), length(t_iceberg)));
        t_non = sort(random_num);
        feature_test(t_iceberg, 3) = 1;
        feature_test(t_non, 3) = 0;
        t_iceberg_s = t_iceberg;
    else
        random_num = t_iceberg(randperm(numel(t_iceberg), length(t_non_all)));
        t_iceberg_s = sort(random_num);
        feature_test(t_iceberg_s, 3) = 1;
        feature_test(t_non_all, 3) = 0;
        t_non = t_non_all;
    end
    
    % Update training data with new samples
    feature_train = [feature_train; feature_test(t_iceberg_s, :); feature_test(t_non, :)];
    p = feature_train(:,3);
    count(j) = length(find(p == 1));
    if count(j) > sample_count + 10
        sample_count = count(j);
    else
        break;
    end
    clear feature_iceberg feature_noniceberg feature_temp;
end

% Map classifier predictions back to the label image
idx = label2idx(L_all);
for i = 1:length(P1_)
    if (weights(1)*P1_(i) + weights(2)*P2_(i) + weights(3)*P3_(i) + weights(4)*P5_(i)) > thres
        t = feature_test(i,1);
        L_res(idx{t}) = 1;
    end
end
end
