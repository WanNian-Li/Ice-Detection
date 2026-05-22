%Author:liuxuying
% Email:liuxuying@usx.edu.cn
% Affiliation:Institute of Artificial Intelligence, Shaoxing University,Shaoxing 312000, China

%Other Co‑authors:Chenzilong
% Email:chenzlong23@mail2.sysu.edu.cn
% Affiliation:School of Geospatial Engineering and Science, Sun Yat-sen University, Southern Marine Science and Engineering
%Last modified date:March 25, 2025
%Explanation:The purpose of this section is to perform image segmentation, applying smoothing and superpixel segmentation operations to the images downloaded from GEE.
%%
clear
clc
iteration=10;
datadir=['HH_202110/'];
filelist=dir([datadir,'*.tif']);
%filename='/Volumes/Liuxy/HH_2018_08_grid_mosaic/grid_121_HH_2018_08.tif';
for p=1:length(filelist)
    tic
    filename=strcat(datadir,filelist(p).name);
    smooth_img=strcat('Smooth_202110/',filelist(p).name(1:19),'_smooth.tif');
    label_img=strcat('Label_202110/',filelist(p).name(1:19),'_label_FS.tif');
    writesliceres(filename,smooth_img,label_img);
    [im_all,R]=geotiffread(filename);
    im_all(isnan(im_all))=-9999;
end
%imshow(imoverlay(im_res,BW,'cyan'),'InitialMagnification',67);










