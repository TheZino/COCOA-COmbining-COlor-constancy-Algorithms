%% Data setup

names = GT{:,1};

mask = 1:length(names);


names = names(mask);

GT = GT{mask,2:4};

for ii = 1:length(methods)
    methods{ii} = methods{ii}{mask,2:4};    
end

switch norm_type
    case 'nG'
        GT= GT ./ GT(:,2);
        
        for ii = 1:length(methods)
            methods{ii} = methods{ii} ./ methods{ii}(:,2);

        end
    case 'nmax'
        GT = GT ./ max(GT,[], 2); 
        for ii = 1:length(methods)
            methods{ii} = methods{ii} ./ max(methods{ii}, [], 2);
        end
    otherwise
        error('normalization type not valid!')
end


%% Data creation

fid = fopen( save_file, 'w' );

for jj = 1 : length(GT)
    
    fprintf( fid, '%s,%d,%d,%d', names{jj}, ...
                                     GT(jj, 1), GT(jj,2), GT(jj,3));
    for ii = 1:length(methods)
        fprintf( fid, ',%d,%d,%d', methods{ii}(jj,1), methods{ii}(jj,2), methods{ii}(jj,3));
    end

    fprintf( fid, '\n');
    
end


fclose( fid );

