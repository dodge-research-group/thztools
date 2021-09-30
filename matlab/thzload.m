function Data = thzload(varargin)
%THZLOAD Recursively loads all terahertz files in a directory
%
%   DATA = THZLOAD recursively loads all files with the .thz extension from
%   the present directory and all of its subdirectories into DATA, an array
%   of DataPulse objects
%
%   DATA = THZLOAD(DIR) loads from the directory DIR
% 
%   DATA = THZLOAD(DIR, EXT) loads all files with the extension EXT
% 
%   DATA = THZLOAD(DIR, EXT, EXCLUDE) excludes directories that contain any
%   strings in the array EXCLUDE (which may also be a character array or a
%   cell array of character vectors)
%
%   DATA = THZLOAD(DIR, EXT, EXCLUDE, DATAIN) appends data loaded from DIR
%   to the input DataPulse array DATAIN
%

p = inputParser;
defaultDir = '.';
defaultExt = 'thz';
defaultExclude = '';
validateExclude = @(x) isstring(x) | ischar(x) | iscellstr(x);
defaultInputData = DataPulse.empty;
addOptional(p,'dirName',defaultDir,@ischar)
addOptional(p,'extension',defaultExt,@ischar)
addOptional(p,'exclude',defaultExclude,validateExclude)
addOptional(p,'InputData',defaultInputData,@(x) isa(x,'DataPulse'))

parse(p,varargin{:});
dirName = p.Results.dirName;
extension = p.Results.extension;
exclude = p.Results.exclude;
InputData = p.Results.InputData;

ListLoc = dir(dirName);
nList = length(ListLoc);
DataLoc = DataPulse.empty;

iDataLoc = 1;
for iFile = 1:nList
    fname = ListLoc(iFile).name;
    if ~ListLoc(iFile).isdir && endsWith(fname,extension)
        DataLoc(iDataLoc) = ...
            DataPulse(fullfile(dirName,fname));
        iDataLoc = iDataLoc+1;
    elseif ListLoc(iFile).isdir ...
            && ~strcmp(fname,'.') ...
            && ~strcmp(fname,'..') ...
            && (~contains(fname,exclude) || isempty(exclude))
        DataLoc = ...
            thzload(fullfile(dirName,fname),extension,exclude,DataLoc);
    end
end
        
Data = [InputData(:);DataLoc(:)];
end

