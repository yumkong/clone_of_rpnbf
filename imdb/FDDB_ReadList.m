function [numItems, filenames] = FDDB_ReadList( listFile)

%{
function [numItems, filenames] = FDDB_ReadList( listFile)
input:
    listFile: the name of the file listing image filenames
output:
    numItems: number of total items contained in the list file
    filenames: cell array of image filenames read from listFile
%}

fin = fopen( listFile, 'rt');
if fin < 0
   filenames = [];
   numItems = -1;
   return;
end

% A = textscan(fin, '%s', '\n'); % in this way filenames containing white
% spaces cannot be correctly handled
% filenams = A{1};

% in this way filenames containing white spaces can be correctly handled
filenames = {};
while ~feof(fin)
   filenames{end+1} = fgetl(fin); 
end

numItems = length(filenames);
fclose(fin);
