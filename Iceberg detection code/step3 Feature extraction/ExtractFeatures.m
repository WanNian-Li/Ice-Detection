%Reassign new labels to the merged label matrix
clc
clear
datadir_Label='Label_202110/';
filelist1=dir([datadir_Label,'*.tif']);
datadir_image='HH_202110/';
filelist2=dir([datadir_image,'*.tif']);
for i=1:length(filelist1)
    name=strsplit(filelist1(i).name,'_');
    temp=name(2);
    file_grid_label(i)=str2num(cell2mat(temp));
end
for i=1:length(filelist2)
    name=strsplit(filelist2(i).name,'_');
    temp=name(2);
    file_grid_im(i)=str2num(cell2mat(temp));
end
for pos=1:length(filelist1)
    filename1=strcat(datadir_Label,filelist1(pos).name); 
    p=find(file_grid_im==file_grid_label(pos));
    filename2=strcat(datadir_image,filelist2(p).name);
    filename=strcat('features_202110/',filelist1(pos).name(1:19),'_features.nc');
    writeFeatureRes(filename1,filename2,filename);
    disp(['done:',filename]);
end