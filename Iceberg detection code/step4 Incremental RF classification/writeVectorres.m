function writeVectorres(L_res, shp_filename, R)
    % Check if the result matrix is empty
    if sum(L_res(:)) == 0
        disp(['abnormal: ', shp_filename]);
        return;
    end

    % Rotate the label image by 90 degrees
    L_res = imrotate(L_res, 90);
    % Fill holes in the binary image
    bw2 = imfill(L_res, 'holes');
    % Remove small objects (area less than 25 pixels)
    bw3 = bwareaopen(bw2, 25);
    % Extract boundaries
    bw = bwboundaries(bw3);
    if isempty(bw)
        disp(['abnormal: ', shp_filename]);
        return;
    end

    num = size(bw, 1);
    STR = 'struct(''Geometry'', values, ''X'', values, ''Y'', values, ''ID'', values)';
    values = cell(num, 1);
    S = eval(STR);
    clear values;

    % Convert pixel coordinates to geographic coordinates
    for i = 1:num
        data = bw{i, 1};
        S(i).Geometry = 'Polygon';
        S(i).ID = i;
        % Convert pixel coordinates (note: row-column conversion)
        data(:,1) = (R.RasterSize(1) - data(:,1));
        [x, y] = intrinsicToGeographic(R, data(:,1), data(:,2));
        S(i).X = y';
        S(i).Y = x';
    end

    % Write the original vector data to a shapefile
    shapewrite(S, shp_filename);
    
    % Generate the .prj file for the WGS84 coordinate system
    prjFile = [shp_filename(1:end-3), 'prj'];
    fid = fopen(prjFile, 'w');
    if fid == -1
        warning('Could not create .prj file: %s', prjFile);
    else
        wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]';
        fprintf(fid, '%s', wkt);
        fclose(fid);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ----- New: Smooth the generated shapefile and overwrite the original file -----
    % Read the shapefile that was just written
    S = shaperead(shp_filename);
    if isempty(S)
        warning('Shapefile %s is empty, skipping smoothing.', shp_filename);
        return;
    end
    
    % Create a new structure array to store the smoothed data
    smoothS = S;
    span = 5;  % Smoothing window size, adjustable as needed
    
    % Iterate through each geometry object to smooth the X and Y coordinates
    for k = 1:length(S)
        % Extract the current object's coordinates
        originalX = S(k).X;
        originalY = S(k).Y;
        % Remove NaN values (shaperead sometimes adds NaN at the end of multiline objects)
        validIdx = ~isnan(originalX) & ~isnan(originalY);
        originalX = originalX(validIdx);
        originalY = originalY(validIdx);
        % Apply the smoothing function using the moving average method
        smoothX = smooth(originalX, span, 'moving');
        smoothY = smooth(originalY, span, 'moving');
        % Assign the smoothed coordinates back to the structure
        smoothS(k).X = smoothX;
        smoothS(k).Y = smoothY;
    end
    
    % Save the smoothed shapefile by overwriting the original file
    shapewrite(smoothS, shp_filename);
    disp(['Smoothed shapefile overwritten: ', shp_filename]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
