%Calculate the inverse distance-weighted mean
function wMean=weightedMean(spx,i,j,cx_temp,cy_temp)
distance=sqrt((i-cx_temp).^2+(j-cy_temp).^2);
distance(distance==0)=1;
d_s=spx.*(1./distance);
wMean=sum(d_s);
end