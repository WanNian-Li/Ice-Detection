%Create a structure to record all superpixel features:
%Row 1: Superpixel label: [Index], Class: 1 - Iceberg, 0 - Non-iceberg, [] - Unclassified
%Row 2: Spatial statistical features: Mean center position, Standard deviation of center position, Inverse distance-weighted mean, Average energy (total energy / number of pixels)
%Row 3: Histogram features: Mean, Variance, Skewness, Kurtosis, Mode, Median, Slope
%Rows 4-6: Entropy, GLCM (Gray Level Co-occurrence Matrix) contrast, correlation, and homogeneity in four directions
%Row 7: Morphological features: Circularity, Compactness, Eccentricity, Range, Orientation
function [res,N]=sample_label2(im_all,L)
idx=label2idx(L);
id=cellfun('length',idx);
idx(id==0)=[];
labelvalues=unique(L);
labelvalues(labelvalues==0)=[];
N=length(labelvalues);
if N==0
    res=[];
end
%Prepare for data allocation for block segmentation
% block=[1:100:N];
% up_edge=block;
% down_edge=block(2:length(block));
% down_edge=down_edge-1;
% down_edge(length(block))=N;
%GLCM (Gray Level Co-occurrence Matrix) calculation parameters
%Calculate morphological features
if N>0
    offset=[0 1;-1 1;-1 0;-1 -1];
    superpx=zeros(N,27);
    for tt=1:N
        value=double(labelvalues(tt));
        pxID=idx{tt};
        spx=im_all(pxID);
        if length(spx)<3
            superpx(tt,:)=0;
            superpx(tt,1)=value;
            superpx(tt,2)=tt;
            superpx(tt,3)=2;
            continue;
        end
        %Row 1: Label value, Index, Class
        class=2;
        [pos(:,1),pos(:,2)]=find(L==value);
        col=max(pos(:,1))-min(pos(:,1))+1;
        row=max(pos(:,2))-min(pos(:,2))+1;
        i=pos(:,1)-min(pos(:,1))+1;
        j=pos(:,2)-min(pos(:,2))+1;
        temp=zeros(col,row);
        temp_L=zeros(col,row);
        for ss=1:length(i)
            temp(i(ss),j(ss))=im_all(pos(ss,1),pos(ss,2));
            temp_L(i(ss),j(ss))=1;
        end
        temp=db2mag(temp);
        stats=regionprops(temp_L,'Centroid');
        %Row 2: Spatial statistical features: Mean center position, Standard deviation of center position, Inverse distance-weighted mean, Average energy (total energy / number of pixels)
        %Mean and Standard Deviation of the center
        c=floor(stats(1).Centroid);
        cx=c(2); cy=c(1);
        if cx~=1 && cx~=size(im_all,1) && cy~=1 && cy~=size(im_all,2)
            cx_temp=cx-min(pos(:,1))+1;
            cy_temp=cy-min(pos(:,2))+1;
            centergrids_3=im_all(cx-1:cx+1,cy-1:cy+1);
            centerMean=mean(centergrids_3(1:9));
            centerStd=std(centergrids_3(1:9));
            weightedMean1=weightedMean(spx,i,j,cx_temp,cy_temp);
        else
            centerMean=im_all(cx,cy);
            centerStd=0;
            weightedMean1=im_all(cx,cy);
        end
        averageEnergy=sum((spx.*spx))/length(spx);
        %Row 3: Histogram features: Mean, Variance, Skewness, Kurtosis, Mode, Median, Slope
        mean1=mean(spx);
        variance=var(spx);
        skewness1=skewness(spx);
        kurtosis1=kurtosis(spx);
        mode1=mode(spx);
        median1=median(spx);
        slope=calslope(spx);
        %Rows 4-6: Texture features
        entropy1=entropy(temp);
        %Calculate GLCM (Gray Level Co-occurrence Matrix)
        [glcm,~]=graycomatrix(temp,'offset',offset);
        res_0=graycoprops(glcm(:,:,1));
        res_45=graycoprops(glcm(:,:,2));
        res_90=graycoprops(glcm(:,:,3));
        res_135=graycoprops(glcm(:,:,4));
        %Save the results of the Gray Level Co-occurrence Matrix (GLCM)
        contrast_0=res_0.Contrast;
        contrast_45=res_45.Contrast;
        contrast_90=res_90.Contrast;
        contrast_135=res_135.Contrast;
        correlation_0=res_0.Correlation;
        correlation_45=res_45.Correlation;
        correlation_90=res_90.Correlation;
        correlation_135=res_135.Correlation;
        homogeneity_0=res_0.Homogeneity;
        homogeneity_45=res_45.Homogeneity;
        homogeneity_90=res_90.Homogeneity;
        homogeneity_135=res_135.Homogeneity;
        %Row 7: Record morphological features
        a=[value tt class centerMean centerStd weightedMean1 averageEnergy mean1 variance skewness1 kurtosis1 mode1 median1 slope entropy1 contrast_0 contrast_45 contrast_90 contrast_135 correlation_0 correlation_45 correlation_90 correlation_135 homogeneity_0 homogeneity_45 homogeneity_90 homogeneity_135];
        a(isnan(a))=0;
        superpx(tt,:)=a;
        clear pos i j spx glcm temp temp_L pxID;
    end
    res=superpx;
    clear superpx;
end
end