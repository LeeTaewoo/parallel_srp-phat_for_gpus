function [filelist]=ld_vadlist()
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

dir= '../1_gen_simul_data/simul_data';

%90
filelist(1,:)={[dir '/090/2m/044cm/reverb430/out0.wav']};

%120
filelist(2,:)={[dir '/120/2m/044cm/reverb430/out0.wav']};

%150
filelist(3,:)={[dir '/150/2m/044cm/reverb430/out0.wav']};

%180
filelist(4,:)={[dir '/180/2m/044cm/reverb430/out0.wav']};
