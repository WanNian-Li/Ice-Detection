%Secondary segmentation for superpixels with multiple peaks in histogram
function Label=furtherSegmentation(im,L)
Label=L*100;
idx=label2idx(L);
id=cellfun('length',idx);
idx(id==0)=[];
labelvalues=unique(L);
labelvalues(labelvalues==0)=[];
N=length(labelvalues);
for ii=1:N
    if length(idx{ii})>1600
        [pos(:,1),pos(:,2)]=find(L==labelvalues(ii));
        col=max(pos(:,1))-min(pos(:,1))+1;
        row=max(pos(:,2))-min(pos(:,2))+1;
        temp=zeros(col,row);
        i=pos(:,1)-min(pos(:,1))+1;
        j=pos(:,2)-min(pos(:,2))+1;
        for ss=1:length(i)
            temp(i(ss),j(ss))=im(pos(ss,1),pos(ss,2));
        end
        tag=ismultipeak(temp);
        if tag==0
            clear pos i j temp;
            continue;
        end
        temp2=db2mag(temp);
        temp2(temp2==1)=0;
        [L2,n]=superpixels(temp2,16);
        idx2=label2idx(L2);
        id2=cellfun('length',idx2);
        idx2(id2==0)=[];
        for jj=1:n
            if mean(im(idx2{jj}))<-15
                L2(idx2{jj})=0;
            end
        end
        for ss=1:length(i)
            Label(pos(ss,1),pos(ss,2))=Label(pos(ss,1),pos(ss,2))+L2(i(ss),j(ss));
        end
        clear pos i j temp temp2 L2 idx2 id2;
    end
end
end