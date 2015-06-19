% Release date: June 2015
% Author: Taewoo Lee, (twlee@speech.korea.ac.kr)
%
% Copyright (C) 2015 Taewoo Lee
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
function []=mk_dir(folder_name);

if exist(folder_name,'dir')
    rmdir(folder_name,'s');
end
mkdir(folder_name);

mkdir([folder_name '/090']);
mkdir([folder_name '/090/2m']);
mkdir([folder_name '/090/2m/044cm']);
mkdir([folder_name '/090/2m/044cm/reverb430']);

mkdir([folder_name '/120']);
mkdir([folder_name '/120/2m']);
mkdir([folder_name '/120/2m/044cm']);
mkdir([folder_name '/120/2m/044cm/reverb430']);

mkdir([folder_name '/150']);
mkdir([folder_name '/150/2m']);
mkdir([folder_name '/150/2m/044cm']);
mkdir([folder_name '/150/2m/044cm/reverb430']);

mkdir([folder_name '/180']);
mkdir([folder_name '/180/2m']);
mkdir([folder_name '/180/2m/044cm']);
mkdir([folder_name '/180/2m/044cm/reverb430']);
