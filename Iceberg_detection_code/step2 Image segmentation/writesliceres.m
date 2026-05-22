function writesliceres(filename,smooth_img,label_img)
 [im_all,R]=geotiffread(filename);
    im_all(isnan(im_all))=-9999;
    %Image Tiling(2000*2000)
    subsize=2000;
    %Image size
    numRows = size(im_all,1);
    numCols = size(im_all,2);
    im_res=zeros(numRows,numCols);
    L_all=zeros(numRows,numCols);
    %Number of Tiles
    sub_nRow=floor(numRows/subsize)+1;
    sub_nCol=floor(numCols/subsize)+1;
    %Boundary Row and Column Indices
    Row_boundary=[1:1:sub_nRow+1];
    Row_boundary=(Row_boundary-1)*subsize+1;
    Row_boundary(sub_nRow+1)=numRows;
    Col_boundary=[1:1:sub_nCol+1];
    Col_boundary=(Col_boundary-1)*subsize+1;
    Col_boundary(sub_nCol+1)=numCols;
    t=1; 
    for i=1:sub_nRow
        for j=1:sub_nCol
             im=im_all(Row_boundary(i):Row_boundary(i+1),Col_boundary(j):Col_boundary(j+1));
%             temp=im;
%             im(all(im<-100,2),:) = [];
%             im(:,all(im<-100,1)) = [];
%             if find(im<-100)<1600
%                 break;
%             end
  
            im_mag_=db2mag(im);
            im_mag=tsmooth(im_mag_);
            im_res(Row_boundary(i):Row_boundary(i+1),Col_boundary(j):Col_boundary(j+1))=im_mag;
            [L1,N1]=superpixels(im_mag,2500,'Compactness',3);
            [L,N]=mergelabel(L1,N1,im_mag);
            L=t*10000+L;
            L1=furtherSegmentation(im_mag,L);
            L_all(Row_boundary(i):Row_boundary(i+1),Col_boundary(j):Col_boundary(j+1))=L1;%Assign block labels as identifiers for differentiation分。
            %disp(['done: Block ',num2str(t)]);
            t=t+1;
        end
    end
    %BW=boundarymask(L_all);
    %im_mag_all=db2mag(im_all);
    im_res=int16(im_res*1000);
    L_all=uint32(L_all);
    geotiffwrite(smooth_img,im_res,R);
    geotiffwrite(label_img,L_all,R);
    disp(['done:',filename]);
end